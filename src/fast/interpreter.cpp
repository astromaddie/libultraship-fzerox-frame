#define NOMINMAX

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <stdio.h>

#include <any>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>
#include <vector>
#include <list>
#include <stack>
#include "fast/resource/type/Light.h"

#ifndef _LANGUAGE_C
#define _LANGUAGE_C
#endif
#include "fast/debug/GfxDebugger.h"
#include "fast/types.h"
#include <string>

#include "fast/interpreter.h"
#include "fast/lus_gbi.h"
#include "fast/backends/gfx_window_manager_api.h"
#include "fast/backends/gfx_rendering_api.h"
#include "fast/backends/gfx_post_shader_pipeline.h"

#include "ship/window/gui/Gui.h"
#include "ship/resource/ResourceManager.h"
#include "ship/utils/Utils.h"
#include "ship/Context.h"
#include "ship/config/ConsoleVariable.h"

#include "libultraship/libultra/os.h"
#include "libultraship/bridge/consolevariablebridge.h"

#include <spdlog/fmt/fmt.h>

#ifdef _WIN32
#include <windows.h>
#endif

std::stack<std::string> currentDir;

// Run-log shim from the port (port_log.h), used by the env-gated diagnostics below.
extern "C" void gdx_dbg_logf(const char* fmt, ...);
extern "C" void gdx_ck(const char* s);
extern "C" void gdx_cki(const char* s, int v);

#define SEG_ADDR(seg, addr) (addr | (seg << 24) | 1)
#define SUPPORT_CHECK(x) assert(x)

// SCALE_M_N: upscale/downscale M-bit integer to N-bit
#define SCALE_5_8(VAL_) (((VAL_)*0xFF) / 0x1F)
#define SCALE_8_5(VAL_) ((((VAL_) + 4) * 0x1F) / 0xFF)
#define SCALE_4_8(VAL_) ((VAL_)*0x11)
#define SCALE_8_4(VAL_) ((VAL_) / 0x11)
#define SCALE_3_8(VAL_) ((VAL_)*0x24)
#define SCALE_8_3(VAL_) ((VAL_) / 0x24)

// Based off the current set native dimensions or active framebuffer
#define HALF_SCREEN_WIDTH(activeFb) ((mFbActive ? activeFb->second.orig_width : mNativeDimensions.width) / 2)
#define HALF_SCREEN_HEIGHT(activeFb) ((mFbActive ? activeFb->second.orig_height : mNativeDimensions.height) / 2)

// Ratios for current window dimensions or active framebuffer scaled size
#define RATIO_X(activeFb, dims) \
    ((mFbActive ? activeFb->second.applied_width : dims.width) / (2.0f * HALF_SCREEN_WIDTH(activeFb)))
#define RATIO_Y(activeFb, dims) \
    ((mFbActive ? activeFb->second.applied_height : dims.height) / (2.0f * HALF_SCREEN_HEIGHT(activeFb)))

#define TEXTURE_CACHE_MAX_SIZE 1024

namespace Fast {

static UcodeHandlers ucode_handler_index = ucode_f3dex2;

const static uint32_t f3dex2AttrHandler[] = {
    F3DEX2_G_MTX_PROJECTION, F3DEX2_G_MTX_LOAD,  F3DEX2_G_MTX_PUSH,  F3DEX_G_MTX_NOPUSH,
    F3DEX2_G_CULL_FRONT,     F3DEX2_G_CULL_BACK, F3DEX2_G_CULL_BOTH,
};

const static uint32_t f3dexAttrHandler[] = { F3DEX_G_MTX_PROJECTION, F3DEX_G_MTX_LOAD,   F3DEX_G_MTX_PUSH,
                                             F3DEX_G_MTX_NOPUSH,     F3DEX_G_CULL_FRONT, F3DEX_G_CULL_BACK,
                                             F3DEX_G_CULL_BOTH };

static constexpr std::array ucode_attr_handlers = {
    &f3dexAttrHandler,  // ucode_f3db
    &f3dexAttrHandler,  // ucode_f3d
    &f3dexAttrHandler,  // ucode_f3dex
    &f3dexAttrHandler,  // ucode_f3exb
    &f3dex2AttrHandler, // ucode_f3ex2
    &f3dex2AttrHandler, // ucode_s2dex
};

static uint32_t get_attr(Attribute attr) {
    const auto ucode_map = ucode_attr_handlers[ucode_handler_index];
    // assert(ucode_map->contains(attr) && "Attribute not found in the current ucode handler");
    return (*ucode_map)[attr];
}

static std::string GetPathWithoutFileName(char* filePath) {
    size_t len = strlen(filePath);

    for (size_t i = len - 1; (long)i >= 0; i--) {
        if (filePath[i] == '/' || filePath[i] == '\\') {
            return std::string(filePath).substr(0, i);
        }
    }

    return filePath;
}

constexpr size_t MAX_TRI_BUFFER = 256;

Interpreter::Interpreter() {
    mRsp = new RSP();
    mRdp = new RDP();
    mBufVbo = new float[MAX_TRI_BUFFER * (32 * 3)];
}

Interpreter::~Interpreter() {
    delete mRsp;
    delete mRdp;
    delete[] mBufVbo;
}

static std::weak_ptr<Interpreter> mInstance;
// Set a cached pointer to the instance so we don't need to go through the window every time
void GfxSetInstance(std::shared_ptr<Interpreter> gfx) {
    mInstance = gfx;
}

// N64 prim_depth is 15-bit (0 near, 0x7FFF far).
static constexpr float N64_PRIM_DEPTH_MAX = 32767.0f;

void Interpreter::Flush() {
    if (mBufVboLen > 0) {
        mGeometryDiagnostics.gpuDrawCalls++;
        mGeometryDiagnostics.gpuTriangles += mBufVboNumTris;
        mRapi->SetCurrentPrimDepth((float)mRdp->prim_depth / N64_PRIM_DEPTH_MAX);
        // G_AC_THRESHOLD compares against the SETBLENDCOLOR alpha register, not a constant.
        mRapi->SetCurrentAlphaCompareThreshold((float)mRdp->blend_color.a / 255.0f);
        mRapi->DrawTriangles(mBufVbo, mBufVboLen, mBufVboNumTris);
        mBufVboLen = 0;
        mBufVboNumTris = 0;
    }
}

// Bit-exact on purpose: a tolerance would hide the sub-ULP drift that flips a triangle
// across a clip plane.
static inline void GdxHashFloat(uint64_t& h, float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    h ^= static_cast<uint64_t>(bits);
    h *= 0x100000001B3ull;
}

// Runs per vertex per pass (~700k hashes/sec in normal play), too expensive to leave on.
static const bool sGdxVertexHashEnabled = [] {
    const char* e = std::getenv("GDX_INTERP_GEO");
    return e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0;
}();

void Interpreter::ResetGeometryDiagnostics() {
    mGeometryDiagnostics = {};
    mGeometryDiagnostics.vertexHash = 0xCBF29CE484222325ull;  // FNV-1a offset basis
    mGeometryDiagnostics.mpFirstHash = 0xCBF29CE484222325ull;
    mGeometryDiagnostics.minNdcX = std::numeric_limits<float>::infinity();
    mGeometryDiagnostics.minNdcY = std::numeric_limits<float>::infinity();
    mGeometryDiagnostics.minNdcZ = std::numeric_limits<float>::infinity();
    mGeometryDiagnostics.maxNdcX = -std::numeric_limits<float>::infinity();
    mGeometryDiagnostics.maxNdcY = -std::numeric_limits<float>::infinity();
    mGeometryDiagnostics.maxNdcZ = -std::numeric_limits<float>::infinity();
}

const GeometryDiagnostics& Interpreter::GetGeometryDiagnostics() const {
    return mGeometryDiagnostics;
}

void Interpreter::SetF3dex2Variant(F3dex2Variant variant) {
    if (mF3dex2Variant != variant) {
        mGeometryDiagnostics.variantSwitches++;
        if (variant == F3dex2Variant::FZeroFlxReject) {
            mGeometryDiagnostics.preFlxVertices = mGeometryDiagnostics.verticesLoaded;
            mGeometryDiagnostics.preFlxTrianglesSubmitted = mGeometryDiagnostics.trianglesSubmitted;
            mGeometryDiagnostics.preFlxTrianglesEmitted = mGeometryDiagnostics.trianglesEmitted;
            mGeometryDiagnostics.preFlxGpuDrawCalls = mGeometryDiagnostics.gpuDrawCalls;
            mGeometryDiagnostics.preFlxGpuTriangles = mGeometryDiagnostics.gpuTriangles;
            mGeometryDiagnostics.preFlxRgba16OpaquePixels = mGeometryDiagnostics.rgba16OpaquePixels;
            mGeometryDiagnostics.preFlxRgba16TransparentPixels = mGeometryDiagnostics.rgba16TransparentPixels;
            mGeometryDiagnostics.preFlxRgba16ForcedOpaquePixels =
                mGeometryDiagnostics.rgba16ForcedOpaquePixels;
            mGeometryDiagnostics.preFlxDepthBypassTriangles = mGeometryDiagnostics.depthBypassTriangles;
            mGeometryDiagnostics.preFlxTexturedTriangles = mGeometryDiagnostics.texturedTriangles;
            mGeometryDiagnostics.preFlxTexture0BoundTriangles = mGeometryDiagnostics.texture0BoundTriangles;
            mGeometryDiagnostics.preFlxTexture0MissingTriangles = mGeometryDiagnostics.texture0MissingTriangles;
            mGeometryDiagnostics.preFlxForcedSimpleMaterialTriangles =
                mGeometryDiagnostics.forcedSimpleMaterialTriangles;
            mGeometryDiagnostics.preFlxShaderId0 = mGeometryDiagnostics.lastShaderId0;
            mGeometryDiagnostics.preFlxShaderId1 = mGeometryDiagnostics.lastShaderId1;
            mGeometryDiagnostics.preFlxTextureWidth = mGeometryDiagnostics.textureWidth;
            mGeometryDiagnostics.preFlxTextureHeight = mGeometryDiagnostics.textureHeight;
            mGeometryDiagnostics.preFlxTextureLineBytes = mGeometryDiagnostics.textureLineBytes;
            mGeometryDiagnostics.preFlxTextureSizeBytes = mGeometryDiagnostics.textureSizeBytes;
            mGeometryDiagnostics.preFlxTextureTmem = mGeometryDiagnostics.textureTmem;
            mGeometryDiagnostics.preFlxTextureTile = mGeometryDiagnostics.textureTile;
            mGeometryDiagnostics.preFlxTextureMaskS = mGeometryDiagnostics.textureMaskS;
            mGeometryDiagnostics.preFlxTextureMaskT = mGeometryDiagnostics.textureMaskT;
            mGeometryDiagnostics.preFlxTextureScaleS = mGeometryDiagnostics.textureScaleS;
            mGeometryDiagnostics.preFlxTextureScaleT = mGeometryDiagnostics.textureScaleT;
            mGeometryDiagnostics.preFlxFogTriangles = mGeometryDiagnostics.fogTriangles;
            mGeometryDiagnostics.preFlxFogBypassTriangles = mGeometryDiagnostics.fogBypassTriangles;
            mGeometryDiagnostics.preFlxMinFogFactor = mGeometryDiagnostics.minFogFactor;
            mGeometryDiagnostics.preFlxMaxFogFactor = mGeometryDiagnostics.maxFogFactor;
            mGeometryDiagnostics.preFlxFogMul = mRsp->fog_mul;
            mGeometryDiagnostics.preFlxFogOffset = mRsp->fog_offset;
            mGeometryDiagnostics.preFlxMinTextureU = mGeometryDiagnostics.minTextureU;
            mGeometryDiagnostics.preFlxMaxTextureU = mGeometryDiagnostics.maxTextureU;
            mGeometryDiagnostics.preFlxMinTextureV = mGeometryDiagnostics.minTextureV;
            mGeometryDiagnostics.preFlxMaxTextureV = mGeometryDiagnostics.maxTextureV;
            mGeometryDiagnostics.preFlxOtherModeH = mRdp->other_mode_h;
            mGeometryDiagnostics.preFlxOtherModeL = mRdp->other_mode_l;
            mGeometryDiagnostics.preFlxCombineMode = mRdp->combine_mode;
            mGeometryDiagnostics.preFlxTexture = mRdp->loaded_texture[0].addr;
        }
        mRsp->f3dflx_alpha_light_valid = false;
    }
    mF3dex2Variant = variant;
}

ShaderProgram* Interpreter::LookupOrCreateShaderProgram(uint64_t id0, uint64_t id1) {
    ShaderProgram* prg = mRapi->LookupShader(id0, id1);
    if (prg == nullptr) {
        mRapi->UnloadShader(mRenderingState.mShaderProgram);
        prg = mRapi->CreateAndLoadNewShader(id0, id1);
        mRenderingState.mShaderProgram = prg;
    }
    return prg;
}

const char* Interpreter::CCMUXtoStr(uint32_t ccmux) {
    static constexpr std::array tbl = {
        "G_CCMUX_COMBINED",
        "G_CCMUX_TEXEL0",
        "G_CCMUX_TEXEL1",
        "G_CCMUX_PRIMITIVE",
        "G_CCMUX_SHADE",
        "G_CCMUX_ENVIRONMENT",
        "G_CCMUX_1",
        "G_CCMUX_COMBINED_ALPHA",
        "G_CCMUX_TEXEL0_ALPHA",
        "G_CCMUX_TEXEL1_ALPHA",
        "G_CCMUX_PRIMITIVE_ALPHA",
        "G_CCMUX_SHADE_ALPHA",
        "G_CCMUX_ENV_ALPHA",
        "G_CCMUX_LOD_FRACTION",
        "G_CCMUX_PRIM_LOD_FRAC",
        "G_CCMUX_K5",
    };
    if (ccmux > tbl.size()) {
        return "G_CCMUX_0";
    }
    return tbl[ccmux];
}

// Seems unused
const char* Interpreter::ACMUXtoStr(uint32_t acmux) {
    static constexpr std::array tbl = {
        "G_ACMUX_COMBINED or G_ACMUX_LOD_FRACTION",
        "G_ACMUX_TEXEL0",
        "G_ACMUX_TEXEL1",
        "G_ACMUX_PRIMITIVE",
        "G_ACMUX_SHADE",
        "G_ACMUX_ENVIRONMENT",
        "G_ACMUX_1 or G_ACMUX_PRIM_LOD_FRAC",
        "G_ACMUX_0",
    };
    return tbl[acmux];
}

void Interpreter::GenerateCC(ColorCombiner* comb, const ColorCombinerKey& key) {
    const bool is2Cyc = (key.options & SHADER_OPT(_2CYC)) != 0;

    uint8_t c[2][2][4];
    uint64_t shaderId0 = 0;
    uint64_t shaderId1 = key.options;
    uint8_t shaderInputMapping[2][7] = { { 0 } };
    bool usedTextures[2]{};
    for (uint32_t i = 0; i < 2 && (i == 0 || is2Cyc); i++) {
        uint32_t rgbA = (key.combine_mode >> (i * 28)) & 0xf;
        uint32_t rgbB = (key.combine_mode >> (i * 28 + 4)) & 0xf;
        uint32_t rgbC = (key.combine_mode >> (i * 28 + 8)) & 0x1f;
        uint32_t rgbD = (key.combine_mode >> (i * 28 + 13)) & 7;
        uint32_t alphaA = (key.combine_mode >> (i * 28 + 16)) & 7;
        uint32_t alphaB = (key.combine_mode >> (i * 28 + 16 + 3)) & 7;
        uint32_t alphaC = (key.combine_mode >> (i * 28 + 16 + 6)) & 7;
        uint32_t alphaD = (key.combine_mode >> (i * 28 + 16 + 9)) & 7;

        if (rgbA >= 8) {
            rgbA = G_CCMUX_0;
        }
        if (rgbB >= 8) {
            rgbB = G_CCMUX_0;
        }
        if (rgbC >= 16) {
            rgbC = G_CCMUX_0;
        }
        if (rgbD == 7) {
            rgbD = G_CCMUX_0;
        }

        if (rgbA == rgbB || rgbC == G_CCMUX_0) {
            // Normalize
            rgbA = G_CCMUX_0;
            rgbB = G_CCMUX_0;
            rgbC = G_CCMUX_0;
        }
        if (alphaA == alphaB || alphaC == G_ACMUX_0) {
            // Normalize
            alphaA = G_ACMUX_0;
            alphaB = G_ACMUX_0;
            alphaC = G_ACMUX_0;
        }
        if (i == 1) {
            if (rgbA != G_CCMUX_COMBINED && rgbB != G_CCMUX_COMBINED && rgbC != G_CCMUX_COMBINED &&
                rgbD != G_CCMUX_COMBINED) {
                // First cycle RGB not used, so clear it away
                c[0][0][0] = c[0][0][1] = c[0][0][2] = c[0][0][3] = G_CCMUX_0;
            }
            if (rgbC != G_CCMUX_COMBINED_ALPHA && alphaA != G_ACMUX_COMBINED && alphaB != G_ACMUX_COMBINED &&
                alphaD != G_ACMUX_COMBINED) {
                // First cycle ALPHA not used, so clear it away
                c[0][1][0] = c[0][1][1] = c[0][1][2] = c[0][1][3] = G_ACMUX_0;
            }
        }

        c[i][0][0] = rgbA;
        c[i][0][1] = rgbB;
        c[i][0][2] = rgbC;
        c[i][0][3] = rgbD;
        c[i][1][0] = alphaA;
        c[i][1][1] = alphaB;
        c[i][1][2] = alphaC;
        c[i][1][3] = alphaD;
    }
    if (!is2Cyc) {
        for (uint32_t i = 0; i < 2; i++) {
            for (uint32_t k = 0; k < 4; k++) {
                c[1][i][k] = i == 0 ? G_CCMUX_0 : G_ACMUX_0;
            }
        }

        // In 1-cycle mode, TEXEL1 returns the same value as TEXEL0.
        // Remap combiner inputs so the shader samples from the correct slot.
        // Ex: TEXEL1/TEXEL1_ALPHA → TEXEL0/TEXEL0_ALPHA
        for (uint32_t k = 0; k < 4; k++) {
            if (c[0][0][k] == G_CCMUX_TEXEL1)
                c[0][0][k] = G_CCMUX_TEXEL0;
            if (c[0][0][k] == G_CCMUX_TEXEL1_ALPHA)
                c[0][0][k] = G_CCMUX_TEXEL0_ALPHA;
            if (c[0][1][k] == G_ACMUX_TEXEL1)
                c[0][1][k] = G_ACMUX_TEXEL0;
        }
    }
    {
        uint8_t inputNumber[32] = { 0 };
        uint32_t nextInputNumber = SHADER_INPUT_1;
        for (uint32_t i = 0; i < 2 && (i == 0 || is2Cyc); i++) {
            for (uint32_t j = 0; j < 4; j++) {
                // Mux values 6/7/15 are overloaded by slot. Only B (value 6 = CENTER, 7 = K4)
                // and C (value 6 = SCALE, 15 = K5) carry chroma-key/convert inputs; A and D
                // reuse those values for unrelated constants.
                if (j == 1 && c[i][0][j] == G_CCMUX_CENTER) {
                    c[i][0][j] = G_CCMUX_KEY_CENTER;
                } else if (j == 1 && c[i][0][j] == G_CCMUX_K4) {
                    c[i][0][j] = G_CCMUX_CONVERT_K4;
                } else if (j == 2 && c[i][0][j] == G_CCMUX_SCALE) {
                    c[i][0][j] = G_CCMUX_KEY_SCALE;
                } else if (j == 2 && c[i][0][j] == G_CCMUX_K5) {
                    c[i][0][j] = G_CCMUX_CONVERT_K5;
                }
                uint32_t val = 0;
                switch (c[i][0][j]) {
                    case G_CCMUX_0:
                        val = SHADER_0;
                        break;
                    case G_CCMUX_1:
                        val = SHADER_1;
                        break;
                    case G_CCMUX_TEXEL0:
                        val = SHADER_TEXEL0;
                        // Set the opposite texture when reading from the second cycle color options
                        if (i == 0) {
                            usedTextures[0] = true;
                        } else {
                            usedTextures[1] = true;
                        }
                        break;
                    case G_CCMUX_TEXEL1:
                        val = SHADER_TEXEL1;
                        if (i == 0) {
                            usedTextures[1] = true;
                        } else {
                            usedTextures[0] = true;
                        }
                        break;
                    case G_CCMUX_TEXEL0_ALPHA:
                        val = SHADER_TEXEL0A;
                        if (i == 0) {
                            usedTextures[0] = true;
                        } else {
                            usedTextures[1] = true;
                        }
                        break;
                    case G_CCMUX_TEXEL1_ALPHA:
                        val = SHADER_TEXEL1A;
                        if (i == 0) {
                            usedTextures[1] = true;
                        } else {
                            usedTextures[0] = true;
                        }
                        break;
                    case G_CCMUX_NOISE:
                        val = SHADER_NOISE;
                        break;
                    case G_CCMUX_PRIMITIVE:
                    case G_CCMUX_PRIMITIVE_ALPHA:
                    case G_CCMUX_PRIM_LOD_FRAC:
                    case G_CCMUX_SHADE:
                    case G_CCMUX_ENVIRONMENT:
                    case G_CCMUX_ENV_ALPHA:
                    case G_CCMUX_LOD_FRACTION:
                    case G_CCMUX_KEY_CENTER:
                    case G_CCMUX_KEY_SCALE:
                    case G_CCMUX_CONVERT_K4:
                    case G_CCMUX_CONVERT_K5:
                        if (inputNumber[c[i][0][j]] == 0) {
                            shaderInputMapping[0][nextInputNumber - 1] = c[i][0][j];
                            inputNumber[c[i][0][j]] = nextInputNumber++;
                        }
                        val = inputNumber[c[i][0][j]];
                        break;
                    case G_CCMUX_COMBINED:
                        val = SHADER_COMBINED;
                        break;
                    default:
                        fprintf(stderr, "Unsupported ccmux: %d\n", c[i][0][j]);
                        break;
                }
                shaderId0 |= (uint64_t)val << (i * 32 + j * 4);
            }
        }
    }
    {
        uint8_t inputNumber[16] = { 0 };
        uint32_t nextInputNumber = SHADER_INPUT_1;
        for (uint32_t i = 0; i < 2; i++) {
            for (uint32_t j = 0; j < 4; j++) {
                uint32_t val = 0;
                switch (c[i][1][j]) {
                    case G_ACMUX_0:
                        val = SHADER_0;
                        break;
                    case G_ACMUX_TEXEL0:
                        val = SHADER_TEXEL0;
                        // Set the opposite texture when reading from the second cycle color options
                        if (i == 0) {
                            usedTextures[0] = true;
                        } else {
                            usedTextures[1] = true;
                        }
                        break;
                    case G_ACMUX_TEXEL1:
                        val = SHADER_TEXEL1;
                        if (i == 0) {
                            usedTextures[1] = true;
                        } else {
                            usedTextures[0] = true;
                        }
                        break;
                    case G_ACMUX_LOD_FRACTION:
                        // case G_ACMUX_COMBINED: same numerical value
                        if (j != 2) {
                            val = SHADER_COMBINED;
                            break;
                        }
                        c[i][1][j] = G_CCMUX_LOD_FRACTION;
                        [[fallthrough]]; // for G_ACMUX_LOD_FRACTION
                    case G_ACMUX_1:
                        // case G_ACMUX_PRIM_LOD_FRAC: same numerical value
                        if (j != 2) {
                            val = SHADER_1;
                            break;
                        }
                        [[fallthrough]]; // for G_ACMUX_PRIM_LOD_FRAC
                    case G_ACMUX_PRIMITIVE:
                    case G_ACMUX_SHADE:
                    case G_ACMUX_ENVIRONMENT:
                        if (inputNumber[c[i][1][j]] == 0) {
                            shaderInputMapping[1][nextInputNumber - 1] = c[i][1][j];
                            inputNumber[c[i][1][j]] = nextInputNumber++;
                        }
                        val = inputNumber[c[i][1][j]];
                        break;
                }
                shaderId0 |= (uint64_t)val << (i * 32 + 16 + j * 4);
            }
        }
    }
    comb->shader_id0 = shaderId0;
    comb->shader_id1 = shaderId1;
    comb->usedTextures[0] = usedTextures[0];
    comb->usedTextures[1] = usedTextures[1];
    // comb->prg = gfx_lookup_or_create_mShaderProgram(shader_id0, shader_id1);
    memcpy(comb->shader_input_mapping, shaderInputMapping, sizeof(shaderInputMapping));
}

ColorCombiner* Interpreter::LookupOrCreateColorCombiner(const ColorCombinerKey& key) {
    if (mPrevCombiner != mColorCombinerPool.end() && mPrevCombiner->first == key) {
        return &mPrevCombiner->second;
    }
    mPrevCombiner = mColorCombinerPool.find(key);
    if (mPrevCombiner != mColorCombinerPool.end()) {
        return &mPrevCombiner->second;
    }
    Flush();
    mPrevCombiner = mColorCombinerPool.insert(std::make_pair(key, ColorCombiner())).first;
    GenerateCC(&mPrevCombiner->second, key);
    return &mPrevCombiner->second;
}

void Interpreter::TextureCacheClear() {
    for (const auto& entry : mTextureCache.map) {
        mTextureCache.free_texture_ids.push_back(entry.second.texture_id);
    }
    mTextureCache.map.clear();
    mTextureCache.lru.clear();
    // Pre-allocate buckets so the map never rehashes during normal operation.
    // Rehashing invalidates all iterators, including those stored in LRU entries.
    mTextureCache.map.reserve(TEXTURE_CACHE_MAX_SIZE);
    // Null rendering-state pointers — they pointed into map nodes that are now freed.
    std::fill(std::begin(mRenderingState.mTextures), std::end(mRenderingState.mTextures), nullptr);
}

// Workshop texture-dump hooks (port/gdx_workshop.cpp).
extern "C" int gdx_workshop_texture_dump_enabled(void);
extern "C" void gdx_workshop_dump_texture(const void* origSrcAddr, size_t origSrcLen,
                                          const char* resourcePathOrNull, const uint8_t* rgba32,
                                          int width, int height, int n64Fmt, int n64Siz);

// Workshop texture packs (port/gdx_workshop.cpp) + loaded-asset registry containing-range lookup
// (port/AssetLoader.cpp), for the per-tile atlas override hook at the LoadBlock/LoadTile
// loaded.addr assignment sites.
extern "C" int gdx_workshop_texture_packs_enabled(void);
extern "C" const char* GdxWorkshopLookupAtlasTileOverride(const char* baseKey, size_t byteOffset, int n64Fmt,
                                                          int n64Siz, int width, int height);
extern "C" const char* GDiffuser_LookupLoadedAssetKeyContaining(const void* addr, size_t* outByteOffset);

void Interpreter::GdxDumpDecodedRgba32(int tile, const uint8_t* rgba32, uint32_t width, uint32_t height) {
    if (!gdx_workshop_texture_dump_enabled()) {
        return;
    }
    uint32_t tmem = mRdp->texture_tile[tile].tmem_index;
    const RawTexMetadata* metadata = &mRdp->loaded_texture[tmem].raw_tex_metadata;
    const void* origAddr = mRdp->loaded_texture[tmem].addr;
    size_t origLen = mRdp->loaded_texture[tmem].size_bytes;
    const char* path = (metadata->resource != nullptr) ? metadata->resource->GetInitData()->Path.c_str() : nullptr;
    gdx_workshop_dump_texture(origAddr, origLen, path, rgba32, (int)width, (int)height,
                              (int)mRdp->texture_tile[tile].fmt, (int)mRdp->texture_tile[tile].siz);
}

void Interpreter::ShaderCacheClear() {
    mRapi->ClearShaderCache();
}

/* Texture-bind signature for testing whether replaying one tick's display list is idempotent:
   interpolation replays the same retained command buffer M times, and StoreLoadedTexture is
   path-dependent, so replay 2 starts from replay 1's end state. */
static uint64_t sGdxTexBindHash = 1469598103934665603ull;

extern "C" void gdx_gfx_texbind_hash_reset(void) {
    sGdxTexBindHash = 1469598103934665603ull;
}

extern "C" unsigned long long gdx_gfx_texbind_hash(void) {
    return (unsigned long long) sGdxTexBindHash;
}

/* Hash the cache key, not texture_id: ids are recycled through free_texture_ids after LRU
   eviction, so identical content can return under a different id. Off by default -- this runs on
   every bind in the hottest loop, replayed M times per tick. */
static const bool sGdxTexBindHashEnabled = [] {
    const char* e = std::getenv("GDX_DIAG_TEXBIND");
    return e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0;
}();

static inline void GdxNoteTexBind(int slot, const TextureCacheKey& key) {
    if (!sGdxTexBindHashEnabled) {
        return;
    }
    const auto mix = [](uint64_t h, uint64_t v) {
        h ^= v;
        return h * 1099511628211ull;
    };
    sGdxTexBindHash = mix(sGdxTexBindHash, (uint64_t) slot);
    sGdxTexBindHash = mix(sGdxTexBindHash, (uint64_t) (uintptr_t) key.texture_addr);
    sGdxTexBindHash = mix(sGdxTexBindHash, (uint64_t) (uintptr_t) key.palette_addrs[0]);
    sGdxTexBindHash = mix(sGdxTexBindHash, (uint64_t) (uintptr_t) key.palette_addrs[1]);
    sGdxTexBindHash = mix(sGdxTexBindHash, ((uint64_t) key.fmt << 8) | key.siz);
    sGdxTexBindHash = mix(sGdxTexBindHash, (uint64_t) key.palette_index);
    sGdxTexBindHash = mix(sGdxTexBindHash, (uint64_t) key.size_bytes);
    sGdxTexBindHash = mix(sGdxTexBindHash, (uint64_t) key.line_size_bytes);
    sGdxTexBindHash = mix(sGdxTexBindHash, ((uint64_t) key.tile_width << 16) | key.tile_height);
}

/* No separate "miss" hash: hit and miss for the same content must contribute
   identically, or a cold-cache pass diverges from a warm one by construction. */

bool Interpreter::TextureCacheLookup(int i, const TextureCacheKey& key) {
    TextureCacheMap::iterator it = mTextureCache.map.find(key);
    TextureCacheNode** n = &mRenderingState.mTextures[i];

    if (it != mTextureCache.map.end()) {
        if (!it->second.uploaded) {
            // A previous decode reserved the slot but bailed before UploadTexture ran, so
            // texture_id has no GPU resource behind it. Treat as a miss and reuse the reserved
            // id rather than binding an empty texture forever.
            static const bool sDiagCiLatch = std::getenv("GDX_DIAG_CI_LATCH") != nullptr;
            if (sDiagCiLatch) {
                static int sNeverUploadedLogs = 0;
                if (sNeverUploadedLogs < 64) {
                    ++sNeverUploadedLogs;
                    SPDLOG_WARN("[ci-latch] refusing never-uploaded cache entry");
                }
            }
            GdxNoteTexBind(i, key);
            mRapi->SelectTexture(i, it->second.texture_id);
            *n = &*it;
            mRenderingState.sampler_valid[i] = true;
            mRenderingState.sampler_linear_filter[i] = false;
            mRenderingState.sampler_cms[i] = 0;
            mRenderingState.sampler_cmt[i] = 0;
            mTextureCache.lru.splice(mTextureCache.lru.end(), mTextureCache.lru,
                                     it->second.lru_location); // move to back
            return false;
        }

        GdxNoteTexBind(i, key);
        mRapi->SelectTexture(i, it->second.texture_id);
        *n = &*it;
        mGeometryDiagnostics.rgba16OpaquePixels += it->second.rgba16_opaque_pixels;
        mGeometryDiagnostics.rgba16TransparentPixels += it->second.rgba16_transparent_pixels;
        mGeometryDiagnostics.rgba16ForcedOpaquePixels += it->second.rgba16_forced_opaque_pixels;
        mTextureCache.lru.splice(mTextureCache.lru.end(), mTextureCache.lru,
                                 it->second.lru_location); // move to back
        return true;
    }

    if (mTextureCache.map.size() >= TEXTURE_CACHE_MAX_SIZE) {
        // Remove the texture that was least recently used
        it = mTextureCache.lru.front().it;
        mTextureCache.free_texture_ids.push_back(it->second.texture_id);
        for (int j = 0; j < SHADER_MAX_TEXTURES; j++) {
            if (mRenderingState.mTextures[j] == &*it)
                mRenderingState.mTextures[j] = nullptr;
        }
        mTextureCache.map.erase(it);
        mTextureCache.lru.pop_front();
    }

    uint32_t texture_id;
    if (!mTextureCache.free_texture_ids.empty()) {
        texture_id = mTextureCache.free_texture_ids.back();
        mTextureCache.free_texture_ids.pop_back();
    } else {
        texture_id = mRapi->NewTexture();
    }

    it = mTextureCache.map.insert(std::make_pair(key, TextureCacheValue())).first;
    TextureCacheNode* node = &*it;
    node->second.texture_id = texture_id;
    node->second.lru_location = mTextureCache.lru.insert(mTextureCache.lru.end(), { it });

    GdxNoteTexBind(i, key);
    mRapi->SelectTexture(i, texture_id);
    mRapi->SetSamplerParameters(i, false, 0, 0);
    mRenderingState.sampler_valid[i] = true;
    mRenderingState.sampler_linear_filter[i] = false;
    mRenderingState.sampler_cms[i] = 0;
    mRenderingState.sampler_cmt[i] = 0;
    *n = node;
    return false;
}

std::string_view Interpreter::GetBaseTexturePath(std::string_view path) {
    if (path.starts_with(Ship::IResource::gAltAssetPrefix)) {
        return path.substr(Ship::IResource::gAltAssetPrefix.length());
    }

    return path;
}

void Interpreter::TextureCacheDelete(const uint8_t* origAddr) {
    while (mTextureCache.map.bucket_count() > 0) {
        TextureCacheKey key = { origAddr, { 0 }, 0, 0, 0 }; // bucket index only depends on the address
        size_t bucket = mTextureCache.map.bucket(key);
        bool again = false;
        for (auto it = mTextureCache.map.begin(bucket); it != mTextureCache.map.end(bucket); ++it) {
            if (it->first.texture_addr == origAddr) {
                for (int j = 0; j < SHADER_MAX_TEXTURES; j++) {
                    if (mRenderingState.mTextures[j] == &*it)
                        mRenderingState.mTextures[j] = nullptr;
                }
                mTextureCache.lru.erase(it->second.lru_location);
                mTextureCache.free_texture_ids.push_back(it->second.texture_id);
                mTextureCache.map.erase(it->first);
                again = true;
                break;
            }
        }
        if (!again) {
            break;
        }
    }

    // The refreshed buffer may also be a TLUT, and palette-keyed entries hash to unrelated
    // buckets. Gated on mSeenPaletteAddrs, so a plain refresh costs one hash lookup.
    TextureCacheDeletePalette(origAddr);
}

// Erase every cache entry whose CI palette key names `paletteAddr`.
//
// CI decodes are keyed on the palette's DRAM address, which cannot see an in-place rewrite of its
// content: the night-course flashing windows restage the same TLUT slot every frame, and
// TextureCacheDelete only matched key.texture_addr. Invalidation rather than content-keying --
// folding palette content into the key (GDX_CI_PALETTE_HASH) froze menu fades.
void Interpreter::TextureCacheDeletePalette(const uint8_t* paletteAddr) {
    if (paletteAddr == nullptr || mSeenPaletteAddrs.find(paletteAddr) == mSeenPaletteAddrs.end()) {
        return;
    }

    static const bool sDiagPaletteEvict = std::getenv("GDX_DIAG_PALETTE_EVICT") != nullptr;
    size_t evicted = 0;

    for (auto it = mTextureCache.map.begin(); it != mTextureCache.map.end();) {
        if (it->first.palette_addrs[0] != paletteAddr && it->first.palette_addrs[1] != paletteAddr) {
            ++it;
            continue;
        }
        for (int j = 0; j < SHADER_MAX_TEXTURES; j++) {
            if (mRenderingState.mTextures[j] == &*it) {
                mRenderingState.mTextures[j] = nullptr;
            }
        }
        mTextureCache.lru.erase(it->second.lru_location);
        mTextureCache.free_texture_ids.push_back(it->second.texture_id);
        it = mTextureCache.map.erase(it);
        ++evicted;
    }

    if (sDiagPaletteEvict && evicted != 0) {
        static int sPaletteEvictLogs = 0;
        if (sPaletteEvictLogs < 200) {
            ++sPaletteEvictLogs;
            SPDLOG_WARN("[pal-evict] palette={} evicted={} cache={}", fmt::ptr(paletteAddr), evicted,
                        mTextureCache.map.size());
        }
    }
}

// Shared ordering counter for the GDX_DIAG_SETTILE / GDX_DIAG_UVPROBE traces so
// tile-descriptor writes and draw markers interleave in one timeline.
int gGdxTraceSeq = 0;
// Set by the port bridge when a race venue is loaded; menu screens would
// otherwise exhaust the probe caps before a race is reached.
extern "C" int gGdxRaceActive = 0;
extern "C" int gGdxVrWorldSpritesLastRun; // defined with the VR sprite marker handling
extern "C" int gGdxVrHudSpritesLastRun;
// Ring buffer of recent tile writes; flushed to disk only when a suspect draw
// fires, so the trace shows exactly the SETTILE history leading to that draw.
#include <deque>
std::deque<std::string> gGdxTileTraceRing;
void GdxTileTracePush(const char* line) {
    gGdxTileTraceRing.emplace_back(line);
    if (gGdxTileTraceRing.size() > 512) {
        gGdxTileTraceRing.pop_front();
    }
}

// Pick the per-line byte width for texture decode. Prefer the DRAM stride from
// loaded_texture when it looks like real per-line info (differs from total size).
// Fall back to the TMEM tile stride when loaded sizes match total (LoadBlock with
// width=1, where line_size == full_image_line_size == size).
static uint32_t GetEffectiveLineSize(uint32_t lineSizeBytes, uint32_t fullImageLineSizeBytes, uint32_t sizeBytes,
                                     uint32_t tileLineSizeBytes) {
    if ((lineSizeBytes != sizeBytes || fullImageLineSizeBytes != sizeBytes) && lineSizeBytes > 0) {
        return lineSizeBytes;
    }
    return tileLineSizeBytes;
}

// Readable-extent probe for TMEM mirror copies: bounds a source read to the
// committed memory region so a lying transfer size can never fault.
static size_t TmemSourceReadableLimit(const uint8_t* addr) {
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi;
    if (VirtualQuery(addr, &mbi, sizeof(mbi)) == sizeof(mbi)) {
        const bool readable = (mbi.State == MEM_COMMIT) &&
                              (mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ |
                                              PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)) != 0;
        if (!readable) {
            return 0;
        }
        const uintptr_t regionEnd = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
        return static_cast<size_t>(regionEnd - reinterpret_cast<uintptr_t>(addr));
    }
#endif
    return SIZE_MAX;
}

// N64 tile masks define the coordinate wrap period independently from the
// amount of source data loaded into TMEM. Uploading the whole backing image
// makes modern samplers wrap at the wrong boundary, which selects adjacent
// mip levels/atlas regions for repeated track and vehicle materials.
static void ApplyTileMaskExtent(const RDP* rdp, int tile, uint32_t& width, uint32_t& height,
                                bool maskAuthoritative = false) {
    const uint8_t masks = rdp->texture_tile[tile].masks;
    const uint8_t maskt = rdp->texture_tile[tile].maskt;
    const uint8_t cms = rdp->texture_tile[tile].cms;
    const uint8_t cmt = rdp->texture_tile[tile].cmt;
    if (masks != G_TX_NOMASK && masks < 31) {
        // On hardware a wrapping axis samples with a period of exactly 1<<mask, regardless of
        // TMEM load bookkeeping. Where the caller can honor that, the mask wins even if the
        // recorded load size implies a smaller extent.
        if (maskAuthoritative && (cms & G_TX_CLAMP) == 0) {
            width = 1u << masks;
        } else {
            width = std::min(width, 1u << masks);
        }
    }
    if (maskt != G_TX_NOMASK && maskt < 31) {
        if (maskAuthoritative && (cmt & G_TX_CLAMP) == 0) {
            height = 1u << maskt;
        } else {
            height = std::min(height, 1u << maskt);
        }
    }
}

static bool IsScaledLinearTexture(const RawTexMetadata& metadata) {
    return metadata.resource != nullptr &&
           (metadata.type == Fast::TextureType::RGBA16bpp ||
            metadata.type == Fast::TextureType::GrayscaleAlpha8bpp ||
            metadata.type == Fast::TextureType::GrayscaleAlpha4bpp) &&
           (metadata.h_byte_scale != 1.0f || metadata.v_pixel_scale != 1.0f);
}

static uint32_t ScaledTextureBitsPerPixel(const RawTexMetadata& metadata) {
    if (metadata.type == Fast::TextureType::GrayscaleAlpha4bpp) {
        return 4;
    }
    return metadata.type == Fast::TextureType::GrayscaleAlpha8bpp ? 8u : 16u;
}

static bool ScaledTextureDimensions(const RawTexMetadata& metadata, uint32_t& sx, uint32_t& sy) {
    const auto& tex = metadata.resource;
    const float h = metadata.h_byte_scale, v = metadata.v_pixel_scale;
    if (tex == nullptr || tex->ImageData == nullptr || !std::isfinite(h) || !std::isfinite(v) ||
        h < 1 || v < 1 || h > 65535 || v > 65535 || std::floor(h) != h || std::floor(v) != v) {
        return false;
    }
    sx = static_cast<uint32_t>(h);
    sy = static_cast<uint32_t>(v);
    return tex->Width > 0 && tex->Height > 0 && tex->Width % sx == 0 && tex->Height % sy == 0 &&
           (uint64_t(tex->Width) * ScaledTextureBitsPerPixel(metadata)) % 8 == 0 &&
           uint64_t(tex->Width) * tex->Height * ScaledTextureBitsPerPixel(metadata) / 8 <= tex->ImageDataSize;
}

static bool SetScaledTextureView(LoadedTexture& loaded, const uint8_t* source, uint32_t x, uint32_t y,
                               uint32_t width, uint32_t height, uint32_t tmemLineBytes) {
    uint32_t sx, sy;
    if (!ScaledTextureDimensions(loaded.raw_tex_metadata, sx, sy) || width == 0 || height == 0) {
        return false;
    }
    const auto& tex = loaded.raw_tex_metadata.resource;
    if (width > tex->Width / sx || height > tex->Height / sy || x >= tex->Width / sx || y >= tex->Height / sy) {
        return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(tex->ImageData);
    const uintptr_t address = reinterpret_cast<uintptr_t>(source);
    const uint32_t pixelBits = ScaledTextureBitsPerPixel(loaded.raw_tex_metadata);
    const uint64_t stride = uint64_t(tex->Width) * pixelBits;
    if (address < base || address - base >= tex->ImageDataSize ||
        ((address - base) * 8 + loaded.resource_bit_offset) % pixelBits != 0) {
        return false;
    }
    const uint64_t offset = (address - base) * 8 + loaded.resource_bit_offset +
                            uint64_t(y) * sy * stride + uint64_t(x) * sx * pixelBits;
    const uint64_t rowBits = uint64_t(width) * sx * pixelBits;
    const uint64_t rowBytes = (rowBits + 7) / 8;
    const uint64_t rows = uint64_t(height) * sy;
    if (rowBits > stride || rows > tex->Height || offset % stride + rowBits > stride ||
        offset + (rows - 1) * stride + rowBits > uint64_t(tex->Width) * tex->Height * pixelBits ||
        rowBytes * rows > UINT32_MAX) {
        return false;
    }
    loaded.addr = tex->ImageData + offset / 8;
    loaded.resource_bit_offset = static_cast<uint8_t>(offset % 8);
    loaded.line_size_bytes = static_cast<uint32_t>(rowBytes);
    loaded.full_image_line_size_bytes = static_cast<uint32_t>(stride / 8);
    loaded.size_bytes = static_cast<uint32_t>(rowBytes * rows);
    loaded.resource_width = width;
    loaded.resource_height = height;
    loaded.resource_tmem_line_bytes = tmemLineBytes;
    return true;
}

static bool ScaledTextureExtent(const RDP* rdp, int tile, uint32_t maxTextureSize,
                              uint32_t& width, uint32_t& height) {
    const auto& tt = rdp->texture_tile[tile];
    const auto& loaded = rdp->loaded_texture[tt.tmem_index];
    uint32_t sx, sy;
    if (!ScaledTextureDimensions(loaded.raw_tex_metadata, sx, sy) || loaded.addr == nullptr) {
        return false;
    }
    if (loaded.raw_tex_metadata.type == Fast::TextureType::GrayscaleAlpha8bpp &&
        (tt.fmt != G_IM_FMT_IA || tt.siz != G_IM_SIZ_8b)) {
        return false;
    }
    if (loaded.raw_tex_metadata.type == Fast::TextureType::GrayscaleAlpha4bpp &&
        (tt.fmt != G_IM_FMT_IA || tt.siz != G_IM_SIZ_4b)) {
        return false;
    }
    width = loaded.resource_width;
    height = loaded.resource_height;
    const uint32_t tileWidth = tt.lrs >= tt.uls ? (tt.lrs - tt.uls + 4) / 4 : 0;
    const uint32_t tileHeight = tt.lrt >= tt.ult ? (tt.lrt - tt.ult + 4) / 4 : 0;
    const uint64_t loadedPixels = uint64_t(width) * height, tilePixels = uint64_t(tileWidth) * tileHeight;
    const bool pyramid = tilePixels > 0 && loadedPixels > tilePixels && loadedPixels * 8 < tilePixels * 13;
    if ((pyramid || (tt.cms & G_TX_CLAMP)) && tileWidth > 0) {
        width = std::min(width, tileWidth);
    }
    if ((pyramid || (tt.cmt & G_TX_CLAMP)) && tileHeight > 0) {
        height = std::min(height, tileHeight);
    }
    ApplyTileMaskExtent(rdp, tile, width, height, true);
    const uint32_t limit = std::min(8192u, maxTextureSize);
    return width > 0 && height > 0 && width <= loaded.resource_width && height <= loaded.resource_height &&
           uint64_t(width) * sx <= limit && uint64_t(height) * sy <= limit;
}

void Interpreter::ImportTextureRgba16(int textureUnit, int tile, bool importReplacement, bool forceOpaqueAlpha) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureRgba16: null texture address for tile {}", tile);
        return;
    }

    // TMEM-emulation decode path (GDX_NO_TMEM=1 disables it): decode from emulated TMEM using
    // only tile-descriptor state, like hardware, so the result does not depend on per-slot load
    // bookkeeping, which goes stale under heavy per-frame TMEM reuse. OTR resources and HD
    // replacements keep the legacy path; their data never enters TMEM.
    static const bool sTmemDisabled = std::getenv("GDX_NO_TMEM") != nullptr;
    const auto& tmemTile = mRdp->texture_tile[tile];
    if (!sTmemDisabled && !importReplacement && metadata->resource == nullptr && tmemTile.siz == G_IM_SIZ_16b) {
        const uint32_t tmemByteOffset = static_cast<uint32_t>(tmemTile.tmem_index) * 8u;
        uint32_t lineBytes = tmemTile.line_size_bytes;
        if (tmemByteOffset < sizeof(mRdp->tmem) && lineBytes >= 2) {
            uint32_t width = lineBytes / 2;
            uint32_t height = (sizeof(mRdp->tmem) - tmemByteOffset) / lineBytes;
            ApplyTileMaskExtent(mRdp, tile, width, height, /*maskAuthoritative=*/true);
            // Clamp axes bound to the declared tile window, like the hardware.
            const uint32_t tileW = static_cast<uint32_t>((tmemTile.lrs - tmemTile.uls + 4) / 4);
            const uint32_t tileH = static_cast<uint32_t>((tmemTile.lrt - tmemTile.ult + 4) / 4);
            if ((tmemTile.cms & G_TX_CLAMP) != 0 && tileW > 0 && tileW < width) {
                width = tileW;
            }
            if ((tmemTile.cmt & G_TX_CLAMP) != 0 && tileH > 0 && tileH < height) {
                height = tileH;
            }
            if (width == 0) {
                width = 1;
            }
            if (height == 0) {
                height = 1;
            }
            if (lineBytes * height > sizeof(mRdp->tmem) - tmemByteOffset) {
                height = (sizeof(mRdp->tmem) - tmemByteOffset) / lineBytes;
            }
            if (width > 0 && height > 0) {
                const uint8_t* tmemSrc = mRdp->tmem + tmemByteOffset;
                uint32_t px = 0;
                uint64_t opaqueCount = 0;
                uint64_t transparentCount = 0;
                uint64_t forcedOpaqueCount = 0;
                for (uint32_t y = 0; y < height; y++) {
                    const uint8_t* row = tmemSrc + static_cast<size_t>(y) * lineBytes;
                    for (uint32_t x = 0; x < width; x++) {
                        const uint16_t col16 = (row[2 * x] << 8) | row[2 * x + 1];
                        uint8_t a = col16 & 1;
                        if (a != 0) {
                            opaqueCount++;
                        } else {
                            transparentCount++;
                            if (forceOpaqueAlpha) {
                                a = 1;
                                forcedOpaqueCount++;
                            }
                        }
                        mTexUploadBuffer[4 * px + 0] = SCALE_5_8(col16 >> 11);
                        mTexUploadBuffer[4 * px + 1] = SCALE_5_8((col16 >> 6) & 0x1f);
                        mTexUploadBuffer[4 * px + 2] = SCALE_5_8((col16 >> 1) & 0x1f);
                        mTexUploadBuffer[4 * px + 3] = a ? 255 : 0;
                        px++;
                    }
                }
                mGeometryDiagnostics.rgba16OpaquePixels += opaqueCount;
                mGeometryDiagnostics.rgba16TransparentPixels += transparentCount;
                mGeometryDiagnostics.rgba16ForcedOpaquePixels += forcedOpaqueCount;
                if (textureUnit >= 0 && textureUnit < SHADER_MAX_TEXTURES &&
                    mRenderingState.mTextures[textureUnit] != nullptr) {
                    TextureCacheValue& cacheValue = mRenderingState.mTextures[textureUnit]->second;
                    cacheValue.rgba16_opaque_pixels = opaqueCount;
                    cacheValue.rgba16_transparent_pixels = transparentCount;
                    cacheValue.rgba16_forced_opaque_pixels = forcedOpaqueCount;
                }
                // A/B: flatten wrapping road-sized textures to gray (also honored
                // here so the diagnostic keeps working on the TMEM decode path).
                static const bool sTmemSolidRoad = std::getenv("GDX_DIAG_SOLIDROAD") != nullptr;
                if (sTmemSolidRoad && width >= 16 && height >= 16 && tmemTile.masks > 0 && tmemTile.maskt > 0) {
                    for (uint32_t p2 = 0; p2 < width * height; ++p2) {
                        mTexUploadBuffer[4 * p2 + 0] = 128;
                        mTexUploadBuffer[4 * p2 + 1] = 128;
                        mTexUploadBuffer[4 * p2 + 2] = 128;
                        mTexUploadBuffer[4 * p2 + 3] = 255;
                    }
                }
                // GDX_DIAG_TMEMCHK=1: log final decode geometry plus a TMEM-vs-DRAM
                // compare, to tell a bad load mirror from bad decode geometry.
                static const bool sTmemChk = std::getenv("GDX_DIAG_TMEMCHK") != nullptr;
                if (sTmemChk && gGdxRaceActive != 0) {
                    static int sTmemChkCount = 0;
                    if (sTmemChkCount < 400) {
                        const auto& slot = mRdp->loaded_texture[tmemTile.tmem_index];
                        uint32_t tmemRowSum = 0;
                        for (uint32_t b = 0; b < lineBytes && b < 256; b++) {
                            tmemRowSum = tmemRowSum * 31u + tmemSrc[b];
                        }
                        int srcCmp = -1; /* -1 = source unavailable */
                        const uint8_t* dram = reinterpret_cast<const uint8_t*>(slot.addr);
                        if (dram != nullptr && slot.size_bytes != 0) {
                            const size_t span = std::min<size_t>(
                                slot.size_bytes, sizeof(mRdp->tmem) - tmemByteOffset);
                            srcCmp = (std::memcmp(tmemSrc, dram, span) == 0) ? 1 : 0;
                        }
                        ++sTmemChkCount;
                        FILE* tf = fopen("tmemchk-trace.txt", sTmemChkCount == 1 ? "w" : "a");
                        if (tf != nullptr) {
                            fprintf(tf,
                                    "D tile=%u tmem=%u line=%u w=%u h=%u masks=%u maskt=%u "
                                    "cms=%u cmt=%u tw=%u th=%u slotsz=%u slotline=%u slotfull=%u "
                                    "rowsum=%08X tmem_eq_dram=%d gen=%u\n",
                                    tile, tmemTile.tmem_index, lineBytes, width, height,
                                    tmemTile.masks, tmemTile.maskt, tmemTile.cms, tmemTile.cmt,
                                    tileW, tileH, slot.size_bytes, slot.line_size_bytes,
                                    slot.full_image_line_size_bytes, tmemRowSum, srcCmp,
                                    mRdp->tmem_generation);
                            fclose(tf);
                        }
                    }
                }
                GdxDumpDecodedRgba32(tile, mTexUploadBuffer, width, height);
                mRapi->UploadTexture(mTexUploadBuffer, width, height);
                return;
            }
        }
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(line_size_bytes, fullImageLineSizeBytes, sizeBytes,
                                               mRdp->texture_tile[tile].line_size_bytes);
    uint32_t width = widthBytes / 2;
    uint32_t height = widthBytes > 0 ? sizeBytes / widthBytes : 0;

    // Preserve the source row stride before reducing the logical upload extent.
    if (fullImageLineSizeBytes == 0 || fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = width * 2;
    }

    // Clamp to the rendered region only when the loaded buffer is ~1.33x of it (mipmap
    // pyramid signature). Window-scrolling tiles have loaded ≈ rendered or loaded >> rendered;
    // skip both. CLAMP wrap mode always opts in.
    uint32_t tile_w = (uint32_t)((mRdp->texture_tile[tile].lrs - mRdp->texture_tile[tile].uls + 4) / 4);
    uint32_t tile_h = (uint32_t)((mRdp->texture_tile[tile].lrt - mRdp->texture_tile[tile].ult + 4) / 4);
    uint32_t loadedPixels = width * height;
    uint32_t renderedPixels = tile_w * tile_h;
    bool pyramidLike =
        renderedPixels > 0 && loadedPixels > renderedPixels && loadedPixels * 8 < renderedPixels * 13; // < 1.625x
    bool clampS = (mRdp->texture_tile[tile].cms & G_TX_CLAMP) != 0;
    bool clampT = (mRdp->texture_tile[tile].cmt & G_TX_CLAMP) != 0;
    if ((pyramidLike || clampS) && tile_w > 0 && tile_w < width) {
        width = tile_w;
    }
    if ((pyramidLike || clampT) && tile_h > 0 && tile_h < height) {
        height = tile_h;
    }
    ApplyTileMaskExtent(mRdp, tile, width, height, /*maskAuthoritative=*/true);

    const bool hdReplacement = !importReplacement && IsScaledLinearTexture(*metadata);
    const bool legacyScaled = !hdReplacement && metadata->resource != nullptr &&
                             (metadata->h_byte_scale != 1.0f || metadata->v_pixel_scale != 1.0f);
    if (hdReplacement) {
        if (!ScaledTextureExtent(mRdp, tile, mRapi->GetMaxTextureSize(), width, height)) {
            return;
        }
        width *= static_cast<uint32_t>(metadata->h_byte_scale);
        height *= static_cast<uint32_t>(metadata->v_pixel_scale);
        fullImageLineSizeBytes = mRdp->loaded_texture[tmemTile.tmem_index].full_image_line_size_bytes;
    }
    const uint32_t rowStrideBytes = (fullImageLineSizeBytes > 0) ? fullImageLineSizeBytes : (width * 2);
    uint32_t readableBytes = hdReplacement
        ? metadata->resource->ImageDataSize - static_cast<uint32_t>(addr - metadata->resource->ImageData)
        : ((rowStrideBytes > 0 && height > 0 && !legacyScaled) ? rowStrideBytes * height : sizeBytes);
#ifdef _WIN32
    {
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery(addr, &mbi, sizeof(mbi)) == sizeof(mbi)) {
            const bool readable = (mbi.State == MEM_COMMIT) &&
                                  (mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY |
                                                  PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                                                  PAGE_EXECUTE_WRITECOPY)) != 0;
            if (!readable) {
                readableBytes = 0;
            } else {
                const uintptr_t regionEnd = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
                const uintptr_t avail = regionEnd - reinterpret_cast<uintptr_t>(addr);
                if (avail < readableBytes) {
                    readableBytes = static_cast<uint32_t>(avail);
                }
            }
        }
    }
#endif
    if (rowStrideBytes > 0 && !hdReplacement) {
        const uint32_t maxRows = readableBytes / rowStrideBytes;
        if (height > maxRows) {
            height = maxRows;
        }
    }
    const uint32_t maxTexel = readableBytes / 2;

    uint32_t i = 0;
    uint64_t opaquePixels = 0;
    uint64_t transparentPixels = 0;
    uint64_t forcedOpaquePixels = 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            const uint32_t srcX = legacyScaled ? static_cast<uint32_t>(x * metadata->h_byte_scale) : x;
            const uint32_t srcY = legacyScaled ? static_cast<uint32_t>(y * metadata->v_pixel_scale) : y;
            uint32_t clrIdx = (srcY * (fullImageLineSizeBytes / 2)) + srcX;

            uint16_t col16 = (clrIdx < maxTexel) ? ((addr[2 * clrIdx] << 8) | addr[2 * clrIdx + 1]) : 0;
            uint8_t a = col16 & 1;
            if (a != 0) {
                opaquePixels++;
            } else {
                transparentPixels++;
                if (forceOpaqueAlpha) {
                    a = 1;
                    forcedOpaquePixels++;
                }
            }
            uint8_t r = col16 >> 11;
            uint8_t g = (col16 >> 6) & 0x1f;
            uint8_t b = (col16 >> 1) & 0x1f;
            mTexUploadBuffer[4 * i + 0] = SCALE_5_8(r);
            mTexUploadBuffer[4 * i + 1] = SCALE_5_8(g);
            mTexUploadBuffer[4 * i + 2] = SCALE_5_8(b);
            mTexUploadBuffer[4 * i + 3] = a ? 255 : 0;

            i++;
        }
    }

    mGeometryDiagnostics.rgba16OpaquePixels += opaquePixels;
    mGeometryDiagnostics.rgba16TransparentPixels += transparentPixels;
    mGeometryDiagnostics.rgba16ForcedOpaquePixels += forcedOpaquePixels;
    if (textureUnit >= 0 && textureUnit < SHADER_MAX_TEXTURES &&
        mRenderingState.mTextures[textureUnit] != nullptr) {
        TextureCacheValue& cacheValue = mRenderingState.mTextures[textureUnit]->second;
        cacheValue.rgba16_opaque_pixels = opaquePixels;
        cacheValue.rgba16_transparent_pixels = transparentPixels;
        cacheValue.rgba16_forced_opaque_pixels = forcedOpaquePixels;
    }

    // A/B: flatten wrapping road-sized textures to gray, to tell whether the track stripes come
    // from the texture or from the geometry/UVs.
    static const bool diagSolidRoad = std::getenv("GDX_DIAG_SOLIDROAD") != nullptr;
    if (diagSolidRoad && width >= 16 && height >= 16 &&
        mRdp->texture_tile[tile].masks > 0 && mRdp->texture_tile[tile].maskt > 0) {
        for (uint32_t p = 0; p < width * height; ++p) {
            mTexUploadBuffer[4 * p + 0] = 128;
            mTexUploadBuffer[4 * p + 1] = 128;
            mTexUploadBuffer[4 * p + 2] = 128;
            mTexUploadBuffer[4 * p + 3] = 255;
        }
    }

    // Dump road-like RGBA16 uploads as PPM images so the actual decoded texture
    // can be inspected directly (coherent road tile vs garbage vs wrong content).
    static const bool diagRgba16 = std::getenv("GDX_DIAG_RGBA16") != nullptr;
    if (diagRgba16 && width > 0 && height > 0) {
        const bool isRoadLike = (height >= 1 && height <= 64 && width >= 16 && width <= 96) &&
                                mRdp->texture_tile[tile].masks > 0 && mRdp->texture_tile[tile].maskt > 0;
        static int sRgba16DumpCount = 0;
        if (isRoadLike && sRgba16DumpCount < 48) {
            char name[128];
            snprintf(name, sizeof(name), "roadtex_%02d_t%u_%ux%u.ppm", sRgba16DumpCount, tile, width, height);
            ++sRgba16DumpCount;
            FILE* f = fopen(name, "wb");
            if (f != nullptr) {
                fprintf(f, "P6\n%u %u\n255\n", width, height);
                for (uint32_t p = 0; p < width * height; ++p) {
                    fputc(mTexUploadBuffer[4 * p + 0], f);
                    fputc(mTexUploadBuffer[4 * p + 1], f);
                    fputc(mTexUploadBuffer[4 * p + 2], f);
                }
                fclose(f);
            }
            FILE* idx = fopen("roadtex-index.txt", "a");
            if (idx != nullptr) {
                const auto& loadedEntry = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index];
                fprintf(idx,
                        "%s fullLine=%u expLine=%u strideOK=%d cms=%u cmt=%u masks=%u maskt=%u scaleS=%04X "
                        "ldLine=%u ldFull=%u ldSize=%u ldOrig=%u tileLine=%u tmemIdx=%u addr=%p\n",
                        name, fullImageLineSizeBytes, width * 2,
                        (fullImageLineSizeBytes == width * 2) ? 1 : 0,
                        mRdp->texture_tile[tile].cms, mRdp->texture_tile[tile].cmt,
                        mRdp->texture_tile[tile].masks, mRdp->texture_tile[tile].maskt,
                        (unsigned)mRsp->texture_scaling_factor.s,
                        loadedEntry.line_size_bytes, loadedEntry.full_image_line_size_bytes,
                        loadedEntry.size_bytes, loadedEntry.orig_size_bytes,
                        mRdp->texture_tile[tile].line_size_bytes,
                        (unsigned)mRdp->texture_tile[tile].tmem_index,
                        (const void*)addr);
                fclose(idx);
            }
        }
    }

    GdxDumpDecodedRgba32(tile, mTexUploadBuffer, width, height);
    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureRgba32(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureRgba32: null texture address for tile {}", tile);
        return;
    }

    uint32_t size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t full_image_line_size_bytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(line_size_bytes, full_image_line_size_bytes, size_bytes,
                                               mRdp->texture_tile[tile].line_size_bytes * 2);
    uint32_t width = widthBytes / 4;
    uint32_t height = widthBytes > 0 ? size_bytes / widthBytes : 0;

    if (full_image_line_size_bytes == 0 || full_image_line_size_bytes == size_bytes) {
        full_image_line_size_bytes = width * 4;
    }

    // Clamp to the rendered region only when the loaded buffer is ~1.33x of it (mipmap
    // pyramid signature). Window-scrolling tiles have loaded ≈ rendered or loaded >> rendered;
    // skip both. CLAMP wrap mode always opts in.
    uint32_t tile_w = (uint32_t)((mRdp->texture_tile[tile].lrs - mRdp->texture_tile[tile].uls + 4) / 4);
    uint32_t tile_h = (uint32_t)((mRdp->texture_tile[tile].lrt - mRdp->texture_tile[tile].ult + 4) / 4);
    uint32_t loadedPixels = width * height;
    uint32_t renderedPixels = tile_w * tile_h;
    bool pyramidLike = renderedPixels > 0 && loadedPixels > renderedPixels && loadedPixels * 8 < renderedPixels * 13;
    bool clampS = (mRdp->texture_tile[tile].cms & G_TX_CLAMP) != 0;
    bool clampT = (mRdp->texture_tile[tile].cmt & G_TX_CLAMP) != 0;
    if ((pyramidLike || clampS) && tile_w > 0 && tile_w < width) {
        width = tile_w;
    }
    if ((pyramidLike || clampT) && tile_h > 0 && tile_h < height) {
        height = tile_h;
    }
    ApplyTileMaskExtent(mRdp, tile, width, height);

    // Copy pixel by pixel, respecting full image stride (handles sub-tile loads)
    uint32_t fullImageStridePixels = full_image_line_size_bytes / 4;
    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t srcIdx = y * fullImageStridePixels + x;
            mTexUploadBuffer[4 * i + 0] = addr[4 * srcIdx + 0];
            mTexUploadBuffer[4 * i + 1] = addr[4 * srcIdx + 1];
            mTexUploadBuffer[4 * i + 2] = addr[4 * srcIdx + 2];
            mTexUploadBuffer[4 * i + 3] = addr[4 * srcIdx + 3];
            i++;
        }
    }
    GdxDumpDecodedRgba32(tile, mTexUploadBuffer, width, height);
    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureIA4(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureIA4: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                               mRdp->texture_tile[tile].line_size_bytes);
    uint32_t width = widthBytes * 2;
    uint32_t height = widthBytes > 0 ? sizeBytes / widthBytes : 0;

    uint32_t firstNibble = 0;
    if (IsScaledLinearTexture(*metadata)) {
        if (!ScaledTextureExtent(mRdp, tile, mRapi->GetMaxTextureSize(), width, height)) {
            return;
        }
        width *= static_cast<uint32_t>(metadata->h_byte_scale);
        height *= static_cast<uint32_t>(metadata->v_pixel_scale);
        firstNibble = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].resource_bit_offset / 4;
    } else {
        if (fullImageLineSizeBytes == 0 || fullImageLineSizeBytes == sizeBytes) {
            fullImageLineSizeBytes = widthBytes;
        }
        ApplyTileMaskExtent(mRdp, tile, width, height);
    }

    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t srcPixelIdx = y * (fullImageLineSizeBytes * 2) + x + firstNibble;
            uint8_t byte = addr[srcPixelIdx / 2];
            uint8_t part = (byte >> (4 - (srcPixelIdx % 2) * 4)) & 0xf;
            uint8_t intensity = part >> 1;
            uint8_t alpha = part & 1;
            mTexUploadBuffer[4 * i + 0] = SCALE_3_8(intensity);
            mTexUploadBuffer[4 * i + 1] = SCALE_3_8(intensity);
            mTexUploadBuffer[4 * i + 2] = SCALE_3_8(intensity);
            mTexUploadBuffer[4 * i + 3] = alpha ? 255 : 0;
            i++;
        }
    }

    GdxDumpDecodedRgba32(tile, mTexUploadBuffer, width, height);
    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureIA8(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureIA8: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t width = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                          mRdp->texture_tile[tile].line_size_bytes);
    uint32_t height = width > 0 ? sizeBytes / width : 0;

    if (IsScaledLinearTexture(*metadata)) {
        if (!ScaledTextureExtent(mRdp, tile, mRapi->GetMaxTextureSize(), width, height)) {
            return;
        }
        width *= static_cast<uint32_t>(metadata->h_byte_scale);
        height *= static_cast<uint32_t>(metadata->v_pixel_scale);
    } else {
        if (fullImageLineSizeBytes == 0 || fullImageLineSizeBytes == sizeBytes) {
            fullImageLineSizeBytes = width;
        }
        ApplyTileMaskExtent(mRdp, tile, width, height);
    }

    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t srcIdx = y * fullImageLineSizeBytes + x;
            uint8_t intensity = addr[srcIdx] >> 4;
            uint8_t alpha = addr[srcIdx] & 0xf;
            mTexUploadBuffer[4 * i + 0] = SCALE_4_8(intensity);
            mTexUploadBuffer[4 * i + 1] = SCALE_4_8(intensity);
            mTexUploadBuffer[4 * i + 2] = SCALE_4_8(intensity);
            mTexUploadBuffer[4 * i + 3] = SCALE_4_8(alpha);
            i++;
        }
    }

    GdxDumpDecodedRgba32(tile, mTexUploadBuffer, width, height);
    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureIA16(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureIA16: null texture address for tile {}", tile);
        return;
    }

    uint32_t size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t full_image_line_size_bytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(line_size_bytes, full_image_line_size_bytes, size_bytes,
                                               mRdp->texture_tile[tile].line_size_bytes);
    uint32_t width = widthBytes / 2;
    uint32_t height = widthBytes > 0 ? size_bytes / widthBytes : 0;

    // A single line of pixels should not equal the entire image (height == 1 non-withstanding)
    if (full_image_line_size_bytes == 0 || full_image_line_size_bytes == size_bytes) {
        full_image_line_size_bytes = width * 2;
    }
    ApplyTileMaskExtent(mRdp, tile, width, height);

    uint32_t i = 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t clrIdx = (y * (full_image_line_size_bytes / 2)) + (x);

            uint8_t intensity = addr[2 * clrIdx];
            uint8_t alpha = addr[2 * clrIdx + 1];
            uint8_t r = intensity;
            uint8_t g = intensity;
            uint8_t b = intensity;
            mTexUploadBuffer[4 * i + 0] = r;
            mTexUploadBuffer[4 * i + 1] = g;
            mTexUploadBuffer[4 * i + 2] = b;
            mTexUploadBuffer[4 * i + 3] = alpha;

            i++;
        }
    }

    GdxDumpDecodedRgba32(tile, mTexUploadBuffer, width, height);
    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureI4(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureI4: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t widthBytes = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                               mRdp->texture_tile[tile].line_size_bytes);
    uint32_t width = widthBytes * 2;
    uint32_t height = widthBytes > 0 ? sizeBytes / widthBytes : 0;

    // A single line of pixels should not equal the entire image (height == 1 non-withstanding)
    if (fullImageLineSizeBytes == 0 || fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = width / 2;
    }

    // Padded I4 assets can carry a wider source row than the logical render tile. Keep the
    // source stride, but crop the upload to the tile, or UV normalization stretches the padding
    // across the glyph and the text reads as narrow and over-spaced.
    const uint32_t tileWidth =
        static_cast<uint32_t>((mRdp->texture_tile[tile].lrs - mRdp->texture_tile[tile].uls + 4) / 4);
    const uint32_t tileHeight =
        static_cast<uint32_t>((mRdp->texture_tile[tile].lrt - mRdp->texture_tile[tile].ult + 4) / 4);
    // TextureUtils uses WRAP even for FONT_SET_4, so clamp bits cannot identify the visible
    // sub-image. The tile bounds are the authoritative upload extent.
    if (tileWidth > 0 && tileWidth < width) {
        width = tileWidth;
    }
    if (tileHeight > 0 && tileHeight < height) {
        height = tileHeight;
    }
    ApplyTileMaskExtent(mRdp, tile, width, height);

    // A render tile whose implied extent exceeds the recorded load walks past the backing buffer
    // (seen as a heap access violation mid-race). Clamp the read to the loaded byte count:
    // bounded garbage plus a reported identity beats a crash.
    if (sizeBytes != 0 && width > 0 && height > 0) {
        const uint64_t lastIdx =
            static_cast<uint64_t>(height - 1) * (fullImageLineSizeBytes * 2) + (width - 1);
        const uint64_t neededBytes = lastIdx / 2 + 1;
        if (neededBytes > sizeBytes) {
            static int sI4ClampLogs = 0;
            if (sI4ClampLogs < 16) {
                ++sI4ClampLogs;
                SPDLOG_ERROR("ImportTextureI4 CLAMP: tile {} extent {}x{} stride {} needs {}B > "
                             "loaded {}B (tmem {} addr {})",
                             tile, width, height, fullImageLineSizeBytes, neededBytes, sizeBytes,
                             mRdp->texture_tile[tile].tmem_index, fmt::ptr(addr));
            }
            const uint32_t rowBytes = fullImageLineSizeBytes != 0 ? fullImageLineSizeBytes : 1u;
            uint32_t maxRows = static_cast<uint32_t>(sizeBytes / rowBytes);
            if (maxRows == 0) {
                maxRows = 1;
            }
            if (height > maxRows) {
                height = maxRows;
            }
        }
    }

    uint32_t i = 0;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t clrIdx = (y * (fullImageLineSizeBytes * 2)) + (x);

            uint8_t byte = addr[clrIdx / 2];
            uint8_t part = (byte >> (4 - (clrIdx % 2) * 4)) & 0xf;
            uint8_t intensity = part;
            uint8_t r = intensity;
            uint8_t g = intensity;
            uint8_t b = intensity;
            uint8_t a = intensity;
            mTexUploadBuffer[4 * i + 0] = SCALE_4_8(r);
            mTexUploadBuffer[4 * i + 1] = SCALE_4_8(g);
            mTexUploadBuffer[4 * i + 2] = SCALE_4_8(b);
            mTexUploadBuffer[4 * i + 3] = SCALE_4_8(a);

            i++;
        }
    }

    /* Checksum the decoded RGBA before upload: zero means the decode produced blackness,
       nonzero-but-invisible convicts the upload/bind layer. */
    {
        static const bool sDiagFontMachine = std::getenv("GDX_DIAG_FONT_MACHINE") != nullptr;
        if (sDiagFontMachine) {
            static int sI4ProbeLogs = 0;
            if (sI4ProbeLogs < 32) {
                ++sI4ProbeLogs;
                uint32_t sum = 0;
                const size_t n = (size_t)width * height * 4;
                for (size_t k = 0; k < n; k++) {
                    sum = sum * 31 + mTexUploadBuffer[k];
                }
                gdx_dbg_logf("[fontmach] I4 decode tile=%d tmem=0x%X w=%u h=%u sum=%08X\n",
                             tile, mRdp->texture_tile[tile].tmem_index, width, height, sum);
                gdx_dbg_logf("[fontmach] I4 in: sizeB=%u origB=%u lineB=%u fullB=%u tileLineB=%u "
                             "tileWH=%ux%u mask=%u/%u cm=%u/%u addr=%p\n",
                             mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes,
                             mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes,
                             lineSizeBytes, fullImageLineSizeBytes, mRdp->texture_tile[tile].line_size_bytes,
                             tileWidth, tileHeight, mRdp->texture_tile[tile].masks, mRdp->texture_tile[tile].maskt,
                             mRdp->texture_tile[tile].cms, mRdp->texture_tile[tile].cmt,
                             static_cast<const void*>(addr));
            }
        }
    }
    GdxDumpDecodedRgba32(tile, mTexUploadBuffer, width, height);
    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureI8(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureI8: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    uint32_t width = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                          mRdp->texture_tile[tile].line_size_bytes);
    uint32_t height = width > 0 ? sizeBytes / width : 0;

    if (fullImageLineSizeBytes == 0 || fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = width;
    }
    ApplyTileMaskExtent(mRdp, tile, width, height);

    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint8_t intensity = addr[y * fullImageLineSizeBytes + x];
            mTexUploadBuffer[4 * i + 0] = intensity;
            mTexUploadBuffer[4 * i + 1] = intensity;
            mTexUploadBuffer[4 * i + 2] = intensity;
            mTexUploadBuffer[4 * i + 3] = intensity;
            i++;
        }
    }

    GdxDumpDecodedRgba32(tile, mTexUploadBuffer, width, height);
    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureCi4(int tile, bool importReplacement) {
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureCi4: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;
    uint32_t palIdx = mRdp->texture_tile[tile].palette; // 0-15

    const uint8_t* palette;

    if (mRdp->palettes[palIdx / 8] == nullptr) {
        SPDLOG_WARN("CI4: null palette slot {} for palIdx={}", palIdx / 8, palIdx);
        return;
    }
    palette = mRdp->palettes[palIdx / 8] + (palIdx % 8) * 16 * 2;

    uint32_t baseLineSizeBytes = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                                      mRdp->texture_tile[tile].line_size_bytes);
    uint32_t resultLineSizeBytes = baseLineSizeBytes;

    if (metadata->h_byte_scale != 1) {
        resultLineSizeBytes *= metadata->h_byte_scale;
    }

    // CI4: 2 pixels per byte
    uint32_t width = resultLineSizeBytes * 2;
    uint32_t height = resultLineSizeBytes > 0 ? sizeBytes / resultLineSizeBytes : 0;

    if (fullImageLineSizeBytes == 0 || fullImageLineSizeBytes == sizeBytes) {
        fullImageLineSizeBytes = resultLineSizeBytes;
    }

    // Clamp to the rendered region only when the loaded buffer is ~1.33x of it (mipmap
    // pyramid signature). Window-scrolling tiles have loaded ≈ rendered or loaded >> rendered;
    // skip both. CLAMP wrap mode always opts in.
    uint32_t tile_w = (uint32_t)((mRdp->texture_tile[tile].lrs - mRdp->texture_tile[tile].uls + 4) / 4);
    uint32_t tile_h = (uint32_t)((mRdp->texture_tile[tile].lrt - mRdp->texture_tile[tile].ult + 4) / 4);
    uint32_t loadedPixels = width * height;
    uint32_t renderedPixels = tile_w * tile_h;
    bool pyramidLike = renderedPixels > 0 && loadedPixels > renderedPixels && loadedPixels * 8 < renderedPixels * 13;
    bool clampS = (mRdp->texture_tile[tile].cms & G_TX_CLAMP) != 0;
    bool clampT = (mRdp->texture_tile[tile].cmt & G_TX_CLAMP) != 0;
    if ((pyramidLike || clampS) && tile_w > 0 && tile_w < width) {
        width = tile_w;
    }
    if ((pyramidLike || clampT) && tile_h > 0 && tile_h < height) {
        height = tile_h;
    }
    ApplyTileMaskExtent(mRdp, tile, width, height);

    uint32_t i = 0;
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t srcPixelIdx = y * (fullImageLineSizeBytes * 2) + x;
            uint8_t byte = addr[srcPixelIdx / 2];
            uint8_t idx = (byte >> (4 - (srcPixelIdx % 2) * 4)) & 0xf;
            uint16_t col16 = (palette[idx * 2] << 8) | palette[idx * 2 + 1]; // Big endian load
            uint8_t a = col16 & 1;
            uint8_t r = col16 >> 11;
            uint8_t g = (col16 >> 6) & 0x1f;
            uint8_t b = (col16 >> 1) & 0x1f;
            mTexUploadBuffer[4 * i + 0] = SCALE_5_8(r);
            mTexUploadBuffer[4 * i + 1] = SCALE_5_8(g);
            mTexUploadBuffer[4 * i + 2] = SCALE_5_8(b);
            mTexUploadBuffer[4 * i + 3] = a ? 255 : 0;
            i++;
        }
    }

    GdxDumpDecodedRgba32(tile, mTexUploadBuffer, width, height);
    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureCi8(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureCi8: null texture address for tile {}", tile);
        return;
    }

    uint32_t sizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t lineSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    if (mRdp->palettes[0] == nullptr || mRdp->palettes[1] == nullptr) {
        SPDLOG_WARN("CI8: null palette (pal0={}, pal1={})", static_cast<const void*>(mRdp->palettes[0]),
                    static_cast<const void*>(mRdp->palettes[1]));
        return;
    }

    uint32_t baseLineSizeBytes = GetEffectiveLineSize(lineSizeBytes, fullImageLineSizeBytes, sizeBytes,
                                                      mRdp->texture_tile[tile].line_size_bytes);
    lineSizeBytes = baseLineSizeBytes;
    if (fullImageLineSizeBytes == 0 || fullImageLineSizeBytes == sizeBytes ||
        fullImageLineSizeBytes < lineSizeBytes) {
        fullImageLineSizeBytes = lineSizeBytes;
    }
    if (lineSizeBytes == 0) {
        SPDLOG_WARN("CI8: zero line size for tile {}", tile);
        return;
    }

    uint32_t resultLineSizeBytes = baseLineSizeBytes;
    if (metadata->h_byte_scale != 1) {
        resultLineSizeBytes *= metadata->h_byte_scale;
    }

    uint32_t width = resultLineSizeBytes;
    uint32_t height = resultLineSizeBytes > 0 ? sizeBytes / resultLineSizeBytes : 0;

    // Clamp to the rendered region only when the loaded buffer is ~1.33x of it (mipmap
    // pyramid signature). Window-scrolling tiles have loaded ≈ rendered or loaded >> rendered;
    // skip both. CLAMP wrap mode always opts in.
    uint32_t tile_w = (uint32_t)((mRdp->texture_tile[tile].lrs - mRdp->texture_tile[tile].uls + 4) / 4);
    uint32_t tile_h = (uint32_t)((mRdp->texture_tile[tile].lrt - mRdp->texture_tile[tile].ult + 4) / 4);
    uint32_t loadedPixels = width * height;
    uint32_t renderedPixels = tile_w * tile_h;
    bool pyramidLike = renderedPixels > 0 && loadedPixels > renderedPixels && loadedPixels * 8 < renderedPixels * 13;
    bool clampS = (mRdp->texture_tile[tile].cms & G_TX_CLAMP) != 0;
    bool clampT = (mRdp->texture_tile[tile].cmt & G_TX_CLAMP) != 0;
    if ((pyramidLike || clampS) && tile_w > 0 && tile_w < width) {
        width = tile_w;
    }
    if ((pyramidLike || clampT) && tile_h > 0 && tile_h < height) {
        height = tile_h;
    }
    ApplyTileMaskExtent(mRdp, tile, width, height);

    uint32_t out = 0;
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            uint8_t idx = addr[y * fullImageLineSizeBytes + x];
            uint16_t col16 = (mRdp->palettes[idx / 128][(idx % 128) * 2] << 8) |
                             mRdp->palettes[idx / 128][(idx % 128) * 2 + 1]; // Big endian load
            uint8_t a = col16 & 1;
            uint8_t r = col16 >> 11;
            uint8_t g = (col16 >> 6) & 0x1f;
            uint8_t b = (col16 >> 1) & 0x1f;
            mTexUploadBuffer[4 * out + 0] = SCALE_5_8(r);
            mTexUploadBuffer[4 * out + 1] = SCALE_5_8(g);
            mTexUploadBuffer[4 * out + 2] = SCALE_5_8(b);
            mTexUploadBuffer[4 * out + 3] = a ? 255 : 0;
            ++out;
        }
    }

    // GDX_MINIMAP_PROBE=1: the minimap uploads as CI8 64x32/64x24 halves using only
    // palette indices 0..3. Logs the index histogram and what the loaded TLUT
    // decodes those entries to (index 1 must be opaque black).
    static const bool sMinimapDecodeProbe = std::getenv("GDX_MINIMAP_PROBE") != nullptr;
    if (sMinimapDecodeProbe && width == 64 && height > 0 && height <= 48) {
        uint32_t hist[256] = { 0 };
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                hist[addr[y * fullImageLineSizeBytes + x]]++;
            }
        }
        uint32_t known = hist[0] + hist[1] + hist[2] + hist[3];
        // Skip generic CI8 rects that merely share the 64-wide dimension.
        if (known * 100 >= (uint32_t)(width * height) * 95) {
            auto dec = [&](uint8_t idx, int& r, int& g, int& b, int& a) {
                uint16_t c = (uint16_t)((mRdp->palettes[idx / 128][(idx % 128) * 2] << 8) |
                                        mRdp->palettes[idx / 128][(idx % 128) * 2 + 1]);
                a = c & 1;
                r = c >> 11;
                g = (c >> 6) & 0x1f;
                b = (c >> 1) & 0x1f;
            };
            int r0, g0, b0, a0, r1, g1, b1, a1, r2, g2, b2, a2, r3, g3, b3, a3;
            dec(0, r0, g0, b0, a0);
            dec(1, r1, g1, b1, a1);
            dec(2, r2, g2, b2, a2);
            dec(3, r3, g3, b3, a3);
            static int sMinimapDecodeLogs = 0;
            if (sMinimapDecodeLogs < 24) {
                ++sMinimapDecodeLogs;
                SPDLOG_ERROR("[minimap-ci8] {}x{} src idx: clear0={} black1={} white2={} grey3={} | "
                             "TLUT pal0=({},{},{},a{}) pal1_BLACK=({},{},{},a{}) pal2=({},{},{},a{}) "
                             "pal3=({},{},{},a{})",
                             width, height, hist[0], hist[1], hist[2], hist[3], r0, g0, b0, a0, r1, g1,
                             b1, a1, r2, g2, b2, a2, r3, g3, b3, a3);
            }
        }
    }

    GdxDumpDecodedRgba32(tile, mTexUploadBuffer, width, height);
    mRapi->UploadTexture(mTexUploadBuffer, width, height);
}

void Interpreter::ImportTextureImg(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureImg: null texture address for tile {}", tile);
        return;
    }

    uint16_t width = metadata->width;
    uint16_t height = metadata->height;
    mRapi->UploadTexture(addr, width, height);
}

void Interpreter::ImportTextureRaw(int tile, bool importReplacement) {
    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* addr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].addr;

    if (addr == nullptr) {
        SPDLOG_ERROR("ImportTextureRaw: null texture address for tile {}", tile);
        return;
    }

    uint16_t width = metadata->width;
    uint16_t height = metadata->height;
    Fast::TextureType type = metadata->type;
    std::shared_ptr<Fast::Texture> resource = metadata->resource;

    // if texture type is CI4 or CI8 we need to apply tlut to it
    switch (type) {
        case Fast::TextureType::Palette4bpp:
            ImportTextureCi4(tile, importReplacement);
            return;
        case Fast::TextureType::Palette8bpp:
            ImportTextureCi8(tile, importReplacement);
            return;
        default:
            break;
    }

    uint32_t numLoadedBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
    uint32_t numOriginallyLoadedBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes;

    uint32_t resultOrigLineSize = mRdp->texture_tile[tile].line_size_bytes;
    switch (mRdp->texture_tile[tile].siz) {
        case G_IM_SIZ_32b:
            resultOrigLineSize *= 2;
            break;
    }
    uint32_t resultOrigHeight = numOriginallyLoadedBytes / resultOrigLineSize;
    uint32_t resultNewLineSize = resultOrigLineSize * metadata->h_byte_scale;
    uint32_t resultNewHeight = resultOrigHeight * metadata->v_pixel_scale;

    if (resultNewLineSize == 4 * width && resultNewHeight == height) {
        // Can use the texture directly since it has the correct dimensions
        mRapi->UploadTexture(addr, width, height);
        return;
    }

    uint32_t fullImageLineSizeBytes =
        mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
    uint32_t line_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;

    // Get the resource's true image size
    uint32_t resourceImageSizeBytes = resource->ImageDataSize;
    uint32_t safeFullImageLineSizeBytes = fullImageLineSizeBytes;
    uint32_t safeLineSizeBytes = line_size_bytes;
    uint32_t safeLoadedBytes = numLoadedBytes;

    // Sometimes the texture load commands will specify a size larger than the authentic texture
    // Normally the OOB info is read as garbage, but will cause a crash on some platforms
    // Restrict the bytes to a safe amount
    if (numLoadedBytes > resourceImageSizeBytes) {
        safeLoadedBytes = resourceImageSizeBytes;
        safeLineSizeBytes = resourceImageSizeBytes;
        safeFullImageLineSizeBytes = resourceImageSizeBytes;
    }

    // Safely only copy the amount of bytes the resource can allow
    for (uint32_t i = 0, j = 0; i < safeLoadedBytes; i += safeLineSizeBytes, j += safeFullImageLineSizeBytes) {
        memcpy(mTexUploadBuffer + i, addr + j, safeLineSizeBytes);
    }

    // Set the remaining bytes to load as 0
    if (numLoadedBytes > resourceImageSizeBytes) {
        memset(mTexUploadBuffer + resourceImageSizeBytes, 0, numLoadedBytes - resourceImageSizeBytes);
    }

    GdxDumpDecodedRgba32(tile, mTexUploadBuffer, resultNewLineSize / 4, resultNewHeight);
    mRapi->UploadTexture(mTexUploadBuffer, resultNewLineSize / 4, resultNewHeight);
}

void Interpreter::ImportTexture(int i, int tile, bool importReplacement) {
    uint8_t fmt = mRdp->texture_tile[tile].fmt;
    uint8_t siz = mRdp->texture_tile[tile].siz;
    uint32_t texFlags = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].tex_flags;
    uint32_t tmemIdex = mRdp->texture_tile[tile].tmem_index;
    uint8_t paletteIndex = mRdp->texture_tile[tile].palette;
    uint32_t origSizeBytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes;

    const RawTexMetadata* metadata = &mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].raw_tex_metadata;
    const uint8_t* origAddr =
        importReplacement && (metadata->resource != nullptr)
            ? mMaskedTextures.find(GetBaseTexturePath(metadata->resource->GetInitData()->Path))->second.replacementData
            : mRdp->loaded_texture[tmemIdex].addr;

    // Lookup-side TMEM trace: pairs with StoreLoadedTexture's store log so one
    // run shows whether this tile's tmem base selected the load that populated
    // it (same env gate as the SETTIMG race trace).
    {
        const bool sTmemTrace = std::getenv("GDX_DIAG_SETTIMG") != nullptr; // live read, see store log
        // Effect tiles 1-4 get their own gate (GDX_DIAG_EFFECTTILE) and budget so
        // this probe cannot be starved by env ordering or the shared log caps.
        static const bool sDiagEffectTile = std::getenv("GDX_DIAG_EFFECTTILE") != nullptr;
        if (sDiagEffectTile && gGdxRaceActive != 0 && tile >= 1 && tile <= 4) {
            static int sEffectTileLogs = 0;
            if (sEffectTileLogs < 64) {
                ++sEffectTileLogs;
                SPDLOG_ERROR("[tmem] EFFECT-TILE lookup tile={} tmem={} fmt={} siz={} -> addr={} sizeB={}",
                             tile, tmemIdex, fmt, siz, fmt::ptr(origAddr),
                             mRdp->loaded_texture[tmemIdex].size_bytes);
            }
        }
        if (sTmemTrace && gGdxRaceActive != 0) { // race-gated, matching the store log
            static int sTmemLookupLogs = 0;
            if (sTmemLookupLogs < 400) {
                ++sTmemLookupLogs;
                SPDLOG_ERROR("[tmem] lookup tile={} tmem={} fmt={} siz={} -> addr={} sizeB={}", tile,
                             tmemIdex, fmt, siz, fmt::ptr(origAddr),
                             mRdp->loaded_texture[tmemIdex].size_bytes);
            }
        }
    }

    // Check if this texture address is a registered GPU framebuffer mirror.
    // If so, bind the GPU FB directly — full resolution, no CPU readback needed.
    if (origAddr != nullptr && !importReplacement) {
        auto fbIt = mFbTextures.find((uintptr_t)origAddr);
        if (fbIt != mFbTextures.end()) {
            Flush();
            mRapi->SelectTextureFb(fbIt->second);
            mRdp->textures_changed[i] = false;
            return;
        }
    }

    if (origAddr == nullptr) {
        static std::array<uint32_t, 32> sNullTextureWarningKeys{};
        static size_t sNullTextureWarningKeyCount = 0;
        const uint32_t warningKey = (static_cast<uint32_t>(tile) << 16) | static_cast<uint16_t>(tmemIdex);
        const auto warningKeysEnd = sNullTextureWarningKeys.begin() + sNullTextureWarningKeyCount;
        if (std::find(sNullTextureWarningKeys.begin(), warningKeysEnd, warningKey) == warningKeysEnd &&
            sNullTextureWarningKeyCount < sNullTextureWarningKeys.size()) {
            sNullTextureWarningKeys[sNullTextureWarningKeyCount++] = warningKey;
            SPDLOG_WARN("ImportTexture: null texture address for tile {} at TMEM word {}; repeats suppressed", tile,
                        tmemIdex);
        }
        return;
    }

    // CI palette-slot guard, hoisted before TextureCacheLookup. mRdp->palettes[] goes transiently
    // null during the port's segment-7 epoch swap at race entry. The old in-decoder check ran
    // after the lookup had already inserted an entry and reserved a texture_id that UploadTexture
    // never filled, so later frames served that empty resource forever. Restricted to siz 4b/8b:
    // CI+16b/32b dispatch to the RGBA importers, which never touch palettes[].
    if (fmt == G_IM_FMT_CI && (siz == G_IM_SIZ_4b || siz == G_IM_SIZ_8b)) {
        bool paletteMissing;
        uint32_t diagPalSlot;
        if (siz == G_IM_SIZ_4b) {
            // Mirrors ImportTextureCi4's derivation: one 16-entry palette; paletteIndex
            // (0-15) selects the staging half via paletteIndex/8.
            diagPalSlot = paletteIndex / 8;
            paletteMissing = mRdp->palettes[diagPalSlot] == nullptr;
        } else {
            // Mirrors ImportTextureCi8's check: CI8 indexes across both staging halves
            // (idx/128 for a 0-255 index), so both halves must be present regardless of
            // paletteIndex.
            diagPalSlot = 0;
            paletteMissing = mRdp->palettes[0] == nullptr || mRdp->palettes[1] == nullptr;
        }
        if (paletteMissing) {
            static const bool sDiagCiLatch = std::getenv("GDX_DIAG_CI_LATCH") != nullptr;
            if (sDiagCiLatch) {
                static int sCiLatchLogs = 0;
                if (sCiLatchLogs < 64) {
                    ++sCiLatchLogs;
                    SPDLOG_WARN("[ci-latch] null palette slot={} palIdx={} addr={} -- decode deferred", diagPalSlot,
                                paletteIndex, fmt::ptr(origAddr));
                }
            }
            return;
        }
    }

    // Use palette_dram_addr (the original DRAM source) instead of palettes[]
    // (which always points to the staging buffer) so the same texture drawn
    // with different palettes gets distinct cache entries.
    TextureCacheKey key;
    if (fmt == G_IM_FMT_CI) {
        if (siz == G_IM_SIZ_4b) {
            uint8_t palSlot = paletteIndex / 8;
            key = { origAddr,
                    { palSlot == 0 ? mRdp->palette_dram_addr[0] : nullptr,
                      palSlot == 1 ? mRdp->palette_dram_addr[1] : nullptr },
                    fmt,
                    siz,
                    paletteIndex,
                    origSizeBytes };
        } else {
            // CI8 uses both palette halves
            key = { origAddr,     { mRdp->palette_dram_addr[0], mRdp->palette_dram_addr[1] }, fmt, siz, paletteIndex,
                    origSizeBytes };
        }
    } else {
        key = { origAddr, {}, fmt, siz, paletteIndex, origSizeBytes };
    }
    key.line_size_bytes = mRdp->loaded_texture[tmemIdex].line_size_bytes;
    key.full_image_line_size_bytes = mRdp->loaded_texture[tmemIdex].full_image_line_size_bytes;

    // CI palette-content hash (opt-in: GDX_CI_PALETTE_HASH). The key carries the palette's DRAM
    // address, which cannot see an in-place fade rewriting content at that same address, so the
    // fade returned a stale decode and froze. Hashes the fixed 256-byte staging buffers, never raw
    // DRAM, so it cannot fault. Off by default, leaving the proven CI path unchanged.
    static const bool sCiPaletteHash = std::getenv("GDX_CI_PALETTE_HASH") != nullptr;
    if (sCiPaletteHash && fmt == G_IM_FMT_CI) {
        uint32_t palHash = 2166136261u;
        const auto foldPalette = [&palHash](const uint8_t* pal) {
            for (uint32_t b = 0; b < 256u; b++) {
                palHash = (palHash ^ pal[b]) * 16777619u;
            }
        };
        if (siz == G_IM_SIZ_4b) {
            // CI4 uses one 16-entry palette; paletteIndex/8 selects the staging half
            // the key's palette_addrs already references.
            foldPalette(mRdp->palette_staging[(paletteIndex / 8) & 1]);
        } else {
            // CI8 spans both palette halves.
            foldPalette(mRdp->palette_staging[0]);
            foldPalette(mRdp->palette_staging[1]);
        }
        key.palette_content_hash = palHash;
    }
    const uint32_t cacheTileWidth =
        mRdp->texture_tile[tile].lrs >= mRdp->texture_tile[tile].uls
            ? (mRdp->texture_tile[tile].lrs - mRdp->texture_tile[tile].uls + 4u) / 4u
            : 0u;
    const uint32_t cacheTileHeight =
        mRdp->texture_tile[tile].lrt >= mRdp->texture_tile[tile].ult
            ? (mRdp->texture_tile[tile].lrt - mRdp->texture_tile[tile].ult + 4u) / 4u
            : 0u;
    key.tile_width = static_cast<uint16_t>(std::min(cacheTileWidth, 0xFFFFu));
    key.tile_height = static_cast<uint16_t>(std::min(cacheTileHeight, 0xFFFFu));
    key.cms = mRdp->texture_tile[tile].cms;
    key.cmt = mRdp->texture_tile[tile].cmt;
    key.masks = mRdp->texture_tile[tile].masks;
    key.maskt = mRdp->texture_tile[tile].maskt;
    key.resource_bit_offset = mRdp->loaded_texture[tmemIdex].resource_bit_offset;
    static const bool diagnosticForcePreFlxOpaqueAlpha =
        std::getenv("GDX_DIAG_FORCE_PREFLX_OPAQUE_ALPHA") != nullptr;
    const bool forceOpaqueAlpha =
        diagnosticForcePreFlxOpaqueAlpha && mF3dex2Variant != F3dex2Variant::FZeroFlxReject;
    key.force_opaque_alpha = forceOpaqueAlpha;

    // TMEM-decoded textures are cached by content: hash the tile's emulated-TMEM span so the same
    // source address with different TMEM contents cannot hit a stale entry. RGBA16 always; the
    // I/IA font paths (which decode reused arena addresses and served stale glyphs) unless
    // GDX_FONT_CONTENT_HASH=0. CI is excluded -- folding it in here froze menu fades, and its
    // palette aspect is handled opt-in above. line_size_bytes already accounts for 4-bit packing,
    // so min(remaining TMEM, lineBytes * 64) covers any format.
    static const bool sTmemCacheDisabled = std::getenv("GDX_NO_TMEM") != nullptr;
    static const bool sFontContentHashDisabled = [] {
        const char* v = std::getenv("GDX_FONT_CONTENT_HASH");
        return v != nullptr && v[0] == '0';
    }();
    const bool isRgba16 = (fmt == G_IM_FMT_RGBA && siz == G_IM_SIZ_16b);
    const bool isFontFmt = (fmt == G_IM_FMT_I && (siz == G_IM_SIZ_4b || siz == G_IM_SIZ_8b)) ||
                           (fmt == G_IM_FMT_IA &&
                            (siz == G_IM_SIZ_4b || siz == G_IM_SIZ_8b || siz == G_IM_SIZ_16b));
    const bool hashThisFormat = isRgba16 || (!sFontContentHashDisabled && isFontFmt);
    if (!sTmemCacheDisabled && !importReplacement && metadata->resource == nullptr && hashThisFormat) {
        const uint32_t tmemByteOffset = static_cast<uint32_t>(tmemIdex) * 8u;
        const uint32_t lineBytes = mRdp->texture_tile[tile].line_size_bytes;
        if (tmemByteOffset < sizeof(mRdp->tmem) && lineBytes >= 2) {
            const uint32_t span =
                std::min<uint32_t>(sizeof(mRdp->tmem) - tmemByteOffset, lineBytes * 64u);
            uint32_t contentHash = 2166136261u;
            const uint8_t* tmemBytes = mRdp->tmem + tmemByteOffset;
            for (uint32_t b = 0; b < span; b++) {
                contentHash = (contentHash ^ tmemBytes[b]) * 16777619u;
            }
            key.tmem_content_hash = contentHash;
        }
    }

    // Zero-sized textures divide by zero or fault UploadTexture. Hoisted before
    // TextureCacheLookup so such a load never mints an uploaded == false entry, which could never
    // be satisfied and re-ran this prologue every frame.
    if (mRdp->texture_tile[tile].line_size_bytes == 0 || mRdp->loaded_texture[tmemIdex].size_bytes == 0 ||
        origAddr == nullptr) {
        return;
    }

    if (!importReplacement && IsScaledLinearTexture(*metadata)) {
        uint32_t width, height;
        if (!ScaledTextureExtent(mRdp, tile, mRapi->GetMaxTextureSize(), width, height)) {
            return;
        }
    }
    if (TextureCacheLookup(i, key)) {
        return;
    }

    // [bg-tex] Race background layer cache miss: skybox (64x1 RGBA16), venue floor/clouds
    // (64x32 RGBA16/IA), and sprites (64x64 CI4). Bounded to avoid drowning the trace.
    if (gGdxRaceActive != 0) {
        static int sBgUploadLogs = 0;
        if (sBgUploadLogs < 48) {
            const auto& tt = mRdp->texture_tile[tile];
            const uint32_t tw = (tt.lrs >= tt.uls) ? (tt.lrs - tt.uls + 4u) / 4u : 0u;
            const uint32_t th = (tt.lrt >= tt.ult) ? (tt.lrt - tt.ult + 4u) / 4u : 0u;
            const bool isBgLike =
                (tw == 64u) &&
                (th == 1u || th == 32u || th == 64u) &&
                ((fmt == G_IM_FMT_RGBA && siz == G_IM_SIZ_16b) ||
                 (fmt == G_IM_FMT_IA && (siz == G_IM_SIZ_8b || siz == G_IM_SIZ_16b)) ||
                 (fmt == G_IM_FMT_CI && siz == G_IM_SIZ_4b));
            if (isBgLike) {
                ++sBgUploadLogs;
                gdx_cki("[bg-tex] upload miss tile", tile);
                gdx_cki("[bg-tex] upload miss fmt_siz", (fmt << 4) | siz);
                gdx_cki("[bg-tex] upload miss wh", (int)((tw << 16) | th));
                gdx_cki("[bg-tex] upload miss addr", (int)(uintptr_t)origAddr);
            }
        }
    }

    if ((texFlags & TEX_FLAG_LOAD_AS_IMG) != 0) {
        ImportTextureImg(tile, importReplacement);
    } else if ((texFlags & TEX_FLAG_LOAD_AS_RAW) != 0) {
        // if load as raw is set then we load_raw();
        ImportTextureRaw(tile, importReplacement);
    } else {
        switch (fmt) {
            case G_IM_FMT_RGBA:
                if (siz == G_IM_SIZ_16b) {
                    ImportTextureRgba16(i, tile, importReplacement, forceOpaqueAlpha);
                } else if (siz == G_IM_SIZ_32b) {
                    ImportTextureRgba32(tile, importReplacement);
                } else {
                    // Rate-limited: no helper runs for this branch, so a persistently
                    // bad dlist would otherwise re-log every frame.
                    static int sRgbaBadSizeLogs = 0;
                    if (sRgbaBadSizeLogs < 8) {
                        ++sRgbaBadSizeLogs;
                        SPDLOG_ERROR("RGBA Texture that isn't 16 or 32 bit. Size = {}", siz);
                    }
                    // OTRTODO: Sometimes, seemingly randomly, we end up here. Could be a bad dlist, could be
                    // something F3D does not have supported. Further investigation is needed.
                }
                break;
            case G_IM_FMT_IA:
                if (siz == G_IM_SIZ_4b) {
                    ImportTextureIA4(tile, importReplacement);
                } else if (siz == G_IM_SIZ_8b) {
                    ImportTextureIA8(tile, importReplacement);
                } else if (siz == G_IM_SIZ_16b) {
                    ImportTextureIA16(tile, importReplacement);
                } else {
                    static int sIaBadSizeLogs = 0;
                    if (sIaBadSizeLogs < 8) {
                        ++sIaBadSizeLogs;
                        SPDLOG_ERROR("IA Texture that isn't 4, 8, or 16 bit. Size = {}", siz);
                    }
                }
                break;
            case G_IM_FMT_CI:
                if (siz == G_IM_SIZ_4b) {
                    ImportTextureCi4(tile, importReplacement);
                } else if (siz == G_IM_SIZ_8b) {
                    ImportTextureCi8(tile, importReplacement);
                } else if (siz == G_IM_SIZ_16b) {
                    // CI+16b is hardware-invalid on N64. The tile's fmt is likely
                    // stale from a prior draw. Decode as RGBA16 instead.
                    ImportTextureRgba16(i, tile, importReplacement, forceOpaqueAlpha);
                } else if (siz == G_IM_SIZ_32b) {
                    ImportTextureRgba32(tile, importReplacement);
                } else {
                    static int sCiBadSizeLogs = 0;
                    if (sCiBadSizeLogs < 8) {
                        ++sCiBadSizeLogs;
                        SPDLOG_ERROR("CI Texture with unexpected size = {}", siz);
                    }
                }
                break;
            case G_IM_FMT_I:
                if (siz == G_IM_SIZ_4b) {
                    ImportTextureI4(tile, importReplacement);
                } else if (siz == G_IM_SIZ_8b) {
                    ImportTextureI8(tile, importReplacement);
                } else {
                    static int sIBadSizeLogs = 0;
                    if (sIBadSizeLogs < 8) {
                        ++sIBadSizeLogs;
                        SPDLOG_ERROR("I Texture that isn't 4 or 8 bit. Size = {}", siz);
                    }
                }
                break;
            case G_IM_FMT_YUV: {
                static int sYuvLogs = 0;
                if (sYuvLogs < 8) {
                    ++sYuvLogs;
                    SPDLOG_ERROR("YUV Textures not supported");
                }
                break;
            }
            default: {
                static int sInvalidFmtLogs = 0;
                if (sInvalidFmtLogs < 8) {
                    ++sInvalidFmtLogs;
                    SPDLOG_ERROR("Invalid texture format. Fmt = {}", fmt);
                }
                break;
            }
        }
    }

    // Mark the entry uploaded even when the dispatch above reached no helper (YUV, unsupported
    // siz/fmt): otherwise the lookup treats it as a permanent miss and the decode re-runs every
    // frame. Safe because the only transient failure, a missing CI palette, returned above before
    // any entry existed.
    if (mRenderingState.mTextures[i] != nullptr) {
        mRenderingState.mTextures[i]->second.uploaded = true;
    }
}

void Interpreter::ImportTextureMask(int i, int tile) {
    uint32_t tmemIndex = mRdp->texture_tile[tile].tmem_index;
    RawTexMetadata metadata = mRdp->loaded_texture[tmemIndex].raw_tex_metadata;

    if (metadata.resource == nullptr) {
        return;
    }

    auto maskIter = mMaskedTextures.find(GetBaseTexturePath(metadata.resource->GetInitData()->Path));
    if (maskIter == mMaskedTextures.end()) {
        return;
    }

    const uint8_t* orig_addr = maskIter->second.mask;

    if (orig_addr == nullptr) {
        return;
    }

    TextureCacheKey key = { orig_addr, {}, 0, 0, 0, 0 };

    if (TextureCacheLookup(i, key)) {
        return;
    }

    uint32_t width = mRdp->texture_tile[tile].line_size_bytes;
    uint32_t height = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes /
                      mRdp->texture_tile[tile].line_size_bytes;
    switch (mRdp->texture_tile[tile].siz) {
        case G_IM_SIZ_4b:
            width *= 2;
            break;
        case G_IM_SIZ_8b:
        default:
            break;
        case G_IM_SIZ_16b:
            width /= 2;
            break;
        case G_IM_SIZ_32b:
            width /= 4;
            break;
    }

    for (uint32_t texIndex = 0; texIndex < width * height; texIndex++) {
        uint8_t masked = orig_addr[texIndex];
        if (masked) {
            mTexUploadBuffer[4 * texIndex + 0] = 0;
            mTexUploadBuffer[4 * texIndex + 1] = 0;
            mTexUploadBuffer[4 * texIndex + 2] = 0;
            mTexUploadBuffer[4 * texIndex + 3] = 0xFF;
        } else {
            mTexUploadBuffer[4 * texIndex + 0] = 0;
            mTexUploadBuffer[4 * texIndex + 1] = 0;
            mTexUploadBuffer[4 * texIndex + 2] = 0;
            mTexUploadBuffer[4 * texIndex + 3] = 0;
        }
    }

    GdxDumpDecodedRgba32(tile, mTexUploadBuffer, width, height);
    mRapi->UploadTexture(mTexUploadBuffer, width, height);
    // This path has its own TextureCacheLookup call, outside ImportTexture's
    // uploaded latch, so latch here or the mask re-decodes every frame.
    if (mRenderingState.mTextures[i] != nullptr) {
        mRenderingState.mTextures[i]->second.uploaded = true;
    }
}

void Interpreter::NormalizeVector(float v[3]) {
    float s = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    v[0] /= s;
    v[1] /= s;
    v[2] /= s;
}

void Interpreter::TransposedMatrixMul(float res[3], const float a[3], const float b[4][4]) {
    res[0] = a[0] * b[0][0] + a[1] * b[0][1] + a[2] * b[0][2];
    res[1] = a[0] * b[1][0] + a[1] * b[1][1] + a[2] * b[1][2];
    res[2] = a[0] * b[2][0] + a[1] * b[2][1] + a[2] * b[2][2];
}

void Interpreter::MatrixMul(float res[4][4], const float a[4][4], const float b[4][4]) {
    float tmp[4][4];
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            tmp[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    memcpy(res, tmp, sizeof(tmp));
}

void Interpreter::CalculateNormalDir(const F3DLight_t* light, float coeffs[3]) {
    float light_dir[3] = { light->dir[0] / 127.0f, light->dir[1] / 127.0f, light->dir[2] / 127.0f };

    Interpreter::TransposedMatrixMul(coeffs, light_dir,
                                     mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1]);
    Interpreter::NormalizeVector(coeffs);
}

void Interpreter::GfxSpMatrix(uint8_t parameters, const int32_t* addr) {
    float matrix[4][4];

    auto it = mCurMtxReplacements->find((Mtx*)addr);
    if (it != mCurMtxReplacements->end()) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                matrix[i][j] = (int)(it->second.mf[i][j] * 65536.0f) * (1.0f / 65536.0f);
            }
        }
    } else {
#ifndef GBI_FLOATS
        // Original GBI where fixed point matrices are used
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j += 2) {
                int32_t int_part = addr[i * 2 + j / 2];
                uint32_t frac_part = addr[8 + i * 2 + j / 2];
                matrix[i][j] = (int32_t)((int_part & 0xffff0000) | (frac_part >> 16)) / 65536.0f;
                matrix[i][j + 1] = (int32_t)((int_part << 16) | (frac_part & 0xffff)) / 65536.0f;
            }
        }
#else
        // For a modified GBI where fixed point values are replaced with floats
        memcpy(matrix, addr, sizeof(matrix));
#endif
    }

    const int8_t mtx_projection = get_attr(MTX_PROJECTION);
    const int8_t mtx_load = get_attr(MTX_LOAD);
    const int8_t mtx_push = get_attr(MTX_PUSH);

    if (parameters & mtx_projection) {
        if (parameters & mtx_load) {
            memcpy(mRsp->P_matrix, matrix, sizeof(matrix));
        } else {
            MatrixMul(mRsp->P_matrix, matrix, mRsp->P_matrix);
        }
    } else { // G_MTX_MODELVIEW
        if ((parameters & mtx_push) && mRsp->modelview_matrix_stack_size < 11) {
            ++mRsp->modelview_matrix_stack_size;
            memcpy(mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1],
                   mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 2], sizeof(matrix));
        }
        if (parameters & mtx_load) {
            if (mRsp->modelview_matrix_stack_size == 0)
                ++mRsp->modelview_matrix_stack_size;
            memcpy(mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1], matrix, sizeof(matrix));
        } else {
            MatrixMul(mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1], matrix,
                      mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1]);
        }
        mRsp->lights_changed = 1;
    }
    MatrixMul(mRsp->MP_matrix, mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1], mRsp->P_matrix);
}

void Interpreter::GfxSpPopMatrix(uint32_t count) {
    while (count--) {
        if (mRsp->modelview_matrix_stack_size > 0) {
            --mRsp->modelview_matrix_stack_size;
            if (mRsp->modelview_matrix_stack_size > 0) {
                MatrixMul(mRsp->MP_matrix, mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1],
                          mRsp->P_matrix);
            }
        }
    }
    mRsp->lights_changed = true;
}

// Runtime fixed-aspect flag, published per frame by the port (input_bridge.c) so EK-editor frames
// render stock 4:3. A plain global, not a CVar: it must never be persisted (a stale saved copy
// pillarboxed the next boot) and it is read on per-frame paths where CVar lookups are too costly.
static int sGdxForceFixedAspect = 0;
extern "C" void gdx_set_force_fixed_aspect(int on) {
    sGdxForceFixedAspect = on ? 1 : 0;
}
extern "C" int gdx_get_force_fixed_aspect(void) {
    return sGdxForceFixedAspect;
}

// Hor+ x-compression factor AdjXForAspectRatio applies to 2D geometry, exposed so the game thread
// can pre-compensate reveal scissors: gDPSetScissor maps linearly across the full frame, but
// gSPTextureRectangle artwork is compressed about screen centre, so a scissor computed in linear
// 320-space clips the panel it should reveal. Reads only per-frame-latched caches, so it is safe
// from the game fiber. Returns 1.0f on 4:3, forced fixed aspect, or widescreen off.
extern "C" float gdx_get_widescreen_geometry_xscale(void) {
    auto gfx = mInstance.lock();
    if (!gfx) {
        return 1.0f;
    }
    if (!gfx->mWidescreenEnabledCache || gfx->mForceFixedAspectCache) {
        return 1.0f;
    }
    if (gfx->mCurDimensions.width == 0 || gfx->mCurDimensions.height == 0) {
        return 1.0f;
    }
    const float aspect = (float)gfx->mCurDimensions.width / (float)gfx->mCurDimensions.height;
    const float xscale = (4.0f / 3.0f) / aspect;
    // The dimension fields are read lock-free from game-logic callers, and a torn read must not
    // leak inf/NaN into callers that cast the result to s32. Floor at 0.15, not 0.25: 48:9
    // triple-monitor yields xscale exactly 0.25.
    if (!(xscale > 0.15f && xscale < 4.0f)) {
        return 1.0f;
    }
    return xscale;
}

// Widening multiplier for the game's CPU-side frustum tests: it culls against +-1.0 in its own
// 4:3 projection space, but under hor+ the visible band is +-(aspect/(4/3)), so on-screen
// ultrawide content was culled (side pop-in). Auto-gated by the current framebuffer aspect: the
// widening factor is exactly 1.0 unless the CVar is on AND the aspect is wider than 16:9, so it
// stays an IEEE-exact no-op on 16:9 and narrower. Separate from the geometry xscale above because
// this one changes culling and stays opt-in.
extern "C" float gdx_get_ultrawide_cull_xscale(void) {
    auto gfx = mInstance.lock();
    if (!gfx || !gfx->mUltrawideCache) {
        return 1.0f;
    }
    if (gfx->mCurDimensions.width == 0 || gfx->mCurDimensions.height == 0) {
        return 1.0f;
    }
    const float aspect = (float)gfx->mCurDimensions.width / (float)gfx->mCurDimensions.height;
    // 16:9 is the stock cull boundary; only widen when the output is meaningfully wider so 16:9
    // users get bit-exact stock behavior even with the CVar enabled.
    if (aspect <= (16.0f / 9.0f) + 0.001f) {
        return 1.0f;
    }
    const float xscale = gdx_get_widescreen_geometry_xscale();
    return (xscale > 0.0f) ? (1.0f / xscale) : 1.0f;
}

float Interpreter::AdjXForAspectRatio(float x) const {
    if (mVrEyeActive && mVrPass != VrPass::HudOnly) {
        return x; // the eye projection already spans the eye's real field of view
    }
    // Skip widescreen adjustment for fixed-size off-screen FBs (HUD elements,
    // small capture buffers), or those which specify a fixed aspect ratio.
    if (mFbActive && mActiveFrameBuffer != mFrameBuffers.end() &&
        (!mActiveFrameBuffer->second.resize || mActiveFrameBuffer->second.forceFixedAspect)) {
        return x;
    } else {
        // Widescreen == 0 skips the hor+ correction so the game renders native 4:3, and
        // Fast3dGui::DrawGame composites into a centred pillarbox. ForceFixedAspect (EK editors)
        // rides the same path. Both are latched per frame in StartFrame, since this runs per
        // vertex where CVar string-hash lookups are too costly.
        if (!mWidescreenEnabledCache || mForceFixedAspectCache) {
            return x;
        }
        return x * (4.0f / 3.0f) / ((float)mCurDimensions.width / (float)mCurDimensions.height);
    }
}

// Scale the width and height value based on the ratio of the viewport to the native size
void Interpreter::AdjustWidthHeightForScale(uint32_t& width, uint32_t& height, uint32_t nativeWidth,
                                            uint32_t nativeHeight) const {
    width = round(width * (mCurDimensions.width / (2.0f * (nativeWidth / 2))));
    height = round(height * (mCurDimensions.height / (2.0f * (nativeHeight / 2))));

    if (width == 0) {
        width = 1;
    }
    if (height == 0) {
        height = 1;
    }
}

void Interpreter::GfxSpVertex(size_t n_vertices, size_t dest_index, const F3DVtx* vertices) {
    if (mVrSuppressDraw && !mVrInHudSprites) {
        return; // nothing is drawn before the HUD marker in a VR HUD pass; skip the transforms
    }
    if (mVrInHudSprites && mVrEyeActive && mVrPass == VrPass::WorldOnly) {
        return;
    }
    for (size_t i = 0; i < n_vertices; i++, dest_index++) {
        const F3DVtx_t* v = &vertices[i].v;
        const F3DVtx_tn* vn = &vertices[i].n;
        struct LoadedVertex* d = &mRsp->loaded_vertices[dest_index];

        if (v == nullptr) {
            return;
        }

        float x = v->ob[0] * mRsp->MP_matrix[0][0] + v->ob[1] * mRsp->MP_matrix[1][0] +
                  v->ob[2] * mRsp->MP_matrix[2][0] + mRsp->MP_matrix[3][0];
        float y = v->ob[0] * mRsp->MP_matrix[0][1] + v->ob[1] * mRsp->MP_matrix[1][1] +
                  v->ob[2] * mRsp->MP_matrix[2][1] + mRsp->MP_matrix[3][1];
        float z = v->ob[0] * mRsp->MP_matrix[0][2] + v->ob[1] * mRsp->MP_matrix[1][2] +
                  v->ob[2] * mRsp->MP_matrix[2][2] + mRsp->MP_matrix[3][2];
        float w = v->ob[0] * mRsp->MP_matrix[0][3] + v->ob[1] * mRsp->MP_matrix[1][3] +
                  v->ob[2] * mRsp->MP_matrix[2][3] + mRsp->MP_matrix[3][3];

        // Fingerprint the transform, not just its cardinality. Capture MP_matrix on the pass's
        // first vertex so an already-divergent matrix is distinguishable from one that drifts
        // partway through the walk.
        if (sGdxVertexHashEnabled) {
            if (mGeometryDiagnostics.verticesLoaded == 0) {
                for (size_t r = 0; r < 4; ++r) {
                    for (size_t c = 0; c < 4; ++c) {
                        GdxHashFloat(mGeometryDiagnostics.mpFirstHash, mRsp->MP_matrix[r][c]);
                    }
                }
            }
            GdxHashFloat(mGeometryDiagnostics.vertexHash, x);
            GdxHashFloat(mGeometryDiagnostics.vertexHash, y);
            GdxHashFloat(mGeometryDiagnostics.vertexHash, z);
            GdxHashFloat(mGeometryDiagnostics.vertexHash, w);
        }

        mGeometryDiagnostics.verticesLoaded++;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(w)) {
            mGeometryDiagnostics.invalidVertices++;
        } else {
            if (w <= 0.0f) {
                mGeometryDiagnostics.verticesNonPositiveW++;
            }
            if (z < -w) {
                mGeometryDiagnostics.verticesOutsideNear++;
            }
            if (z > w) {
                mGeometryDiagnostics.verticesOutsideFar++;
            }
            if (fabsf(w) >= 0.000001f) {
                const float ndcX = x / w;
                const float ndcY = y / w;
                const float ndcZ = z / w;
                mGeometryDiagnostics.minNdcX = std::min(mGeometryDiagnostics.minNdcX, ndcX);
                mGeometryDiagnostics.maxNdcX = std::max(mGeometryDiagnostics.maxNdcX, ndcX);
                mGeometryDiagnostics.minNdcY = std::min(mGeometryDiagnostics.minNdcY, ndcY);
                mGeometryDiagnostics.maxNdcY = std::max(mGeometryDiagnostics.maxNdcY, ndcY);
                mGeometryDiagnostics.minNdcZ = std::min(mGeometryDiagnostics.minNdcZ, ndcZ);
                mGeometryDiagnostics.maxNdcZ = std::max(mGeometryDiagnostics.maxNdcZ, ndcZ);
            }
        }

        float world_pos[3] = { 0.0 };
        if (mRsp->geometry_mode & G_LIGHTING_POSITIONAL) {
            float(*mtx)[4] = mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1];
            world_pos[0] = v->ob[0] * mtx[0][0] + v->ob[1] * mtx[1][0] + v->ob[2] * mtx[2][0] + mtx[3][0];
            world_pos[1] = v->ob[0] * mtx[0][1] + v->ob[1] * mtx[1][1] + v->ob[2] * mtx[2][1] + mtx[3][1];
            world_pos[2] = v->ob[0] * mtx[0][2] + v->ob[1] * mtx[1][2] + v->ob[2] * mtx[2][2] + mtx[3][2];
        }

        // Honor G_EX_WIDESCREEN_STRETCH here as well as in GfxDrawRectangle: screen transitions
        // redraw the captured widescreen frame as triangles inside a STRETCH scope, and the plain
        // hor+ path would squeeze that capture into the central 4:3 band. Scaling clip-space x
        // scales NDC x by the same factor, so this is exactly the rectangle path's adjustment.
        // With the flag unset the historical single AdjXForAspectRatio call runs unchanged.
        if ((mRsp->extra_geometry_mode & G_EX_WIDESCREEN_STRETCH) != 0) {
            const bool fixedAspectFramebuffer =
                mFbActive && mActiveFrameBuffer != mFrameBuffers.end() &&
                (!mActiveFrameBuffer->second.resize || mActiveFrameBuffer->second.forceFixedAspect);
            const bool widescreenFrameActive =
                !fixedAspectFramebuffer && !(mVrEyeActive && mVrPass != VrPass::HudOnly) &&
                ((float)mCurDimensions.width / (float)mCurDimensions.height) > (4.0f / 3.0f) &&
                mWidescreenEnabledCache && !mForceFixedAspectCache;
            if (widescreenFrameActive) {
                // Identical calibration to GfxDrawRectangle: native safe area x=12..308 -> full viewport.
                constexpr float kSafeAreaScale = 160.0f / (160.0f - 12.0f);
                x = x * kSafeAreaScale;
            } else {
                x = AdjXForAspectRatio(x);
            }
        } else {
            x = AdjXForAspectRatio(x);
        }

        short U = v->tc[0] * mRsp->texture_scaling_factor.s >> 16;
        short V = v->tc[1] * mRsp->texture_scaling_factor.t >> 16;
        bool hasF3dflxAlpha = false;
        uint8_t f3dflxAlpha = 0;

        if (mRsp->geometry_mode & G_LIGHTING) {
            if (mRsp->lights_changed) {
                for (int i = 0; i < mRsp->current_num_lights - 1; i++) {
                    CalculateNormalDir(&mRsp->current_lights[i].l, mRsp->current_lights_coeffs[i]);
                }
                /*static const Light_t lookat_x = {{0, 0, 0}, 0, {0, 0, 0}, 0, {127, 0, 0}, 0};
                static const Light_t lookat_y = {{0, 0, 0}, 0, {0, 0, 0}, 0, {0, 127, 0}, 0};*/
                CalculateNormalDir(&mRsp->lookat[0], mRsp->current_lookat_coeffs[0]);
                CalculateNormalDir(&mRsp->lookat[1], mRsp->current_lookat_coeffs[1]);
                mRsp->lights_changed = false;
            }

            int r = mRsp->current_lights[mRsp->current_num_lights - 1].l.col[0];
            int g = mRsp->current_lights[mRsp->current_num_lights - 1].l.col[1];
            int b = mRsp->current_lights[mRsp->current_num_lights - 1].l.col[2];

            for (int i = 0; i < mRsp->current_num_lights - 1; i++) {
                float intensity = 0;
                if ((mRsp->geometry_mode & G_LIGHTING_POSITIONAL) && (mRsp->current_lights[i].p.unk3 != 0)) {
                    // Calculate distance from the light to the vertex
                    float dist_vec[3] = { mRsp->current_lights[i].p.pos[0] - world_pos[0],
                                          mRsp->current_lights[i].p.pos[1] - world_pos[1],
                                          mRsp->current_lights[i].p.pos[2] - world_pos[2] };
                    float dist_sq =
                        dist_vec[0] * dist_vec[0] + dist_vec[1] * dist_vec[1] +
                        dist_vec[2] * dist_vec[2] * 2; // The *2 comes from GLideN64, unsure of why it does it
                    float dist = sqrt(dist_sq);

                    // Transform distance vector (which acts as a direction light vector) into model's space
                    float light_model[3];
                    TransposedMatrixMul(light_model, dist_vec,
                                        mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1]);

                    // Calculate intensity for each axis using standard formula for intensity
                    float light_intensity[3];
                    for (int light_i = 0; light_i < 3; light_i++) {
                        light_intensity[light_i] = 4.0f * light_model[light_i] / dist_sq;
                        light_intensity[light_i] = std::clamp(light_intensity[light_i], -1.0f, 1.0f);
                    }

                    // Adjust intensity based on surface normal and sum up total
                    float total_intensity =
                        light_intensity[0] * vn->n[0] + light_intensity[1] * vn->n[1] + light_intensity[2] * vn->n[2];
                    total_intensity = std::clamp(total_intensity, -1.0f, 1.0f);

                    // Attenuate intensity based on attenuation values.
                    // Example formula found at https://ogldev.org/www/tutorial20/tutorial20.html
                    // Specific coefficients for MM's microcode sourced from GLideN64
                    // https://github.com/gonetz/GLideN64/blob/3b43a13a80dfc2eb6357673440b335e54eaa3896/src/gSP.cpp#L636
                    float distf = floorf(dist);
                    float attenuation = (distf * mRsp->current_lights[i].p.unk7 * 2.0f +
                                         distf * distf * mRsp->current_lights[i].p.unkE / 8.0f) /
                                            (float)0xFFFF +
                                        1.0f;
                    intensity = total_intensity / attenuation;
                } else {
                    intensity += vn->n[0] * mRsp->current_lights_coeffs[i][0];
                    intensity += vn->n[1] * mRsp->current_lights_coeffs[i][1];
                    intensity += vn->n[2] * mRsp->current_lights_coeffs[i][2];
                    intensity /= 127.0f;
                }
                if (intensity > 0.0f) {
                    r += intensity * mRsp->current_lights[i].l.col[0];
                    g += intensity * mRsp->current_lights[i].l.col[1];
                    b += intensity * mRsp->current_lights[i].l.col[2];
                }
            }

            d->color.r = r > 255 ? 255 : r;
            d->color.g = g > 255 ? 255 : g;
            d->color.b = b > 255 ? 255 : b;

            if (mRsp->geometry_mode & G_TEXTURE_GEN) {
                if (mF3dex2Variant == F3dex2Variant::FZeroFlxReject &&
                    mRsp->f3dflx_alpha_light_valid) {
                    float alphaLight[3];
                    TransposedMatrixMul(alphaLight, mRsp->f3dflx_alpha_light,
                                        mRsp->modelview_matrix_stack[mRsp->modelview_matrix_stack_size - 1]);
                    NormalizeVector(alphaLight);
                    const float intensity = alphaLight[0] * vn->n[0] + alphaLight[1] * vn->n[1] +
                                            alphaLight[2] * vn->n[2];
                    const int lutIndex = std::clamp(static_cast<int>(intensity), -128, 127);
                    const int dmemIndex = static_cast<int>(mRsp->dma_io_dmem) + 128 + lutIndex;
                    if (dmemIndex >= 0 && dmemIndex < static_cast<int>(sizeof(mRsp->dmem))) {
                        f3dflxAlpha = mRsp->dmem[dmemIndex];
                        hasF3dflxAlpha = true;
                        mGeometryDiagnostics.f3dflxAlphaVertices++;
                    }
                } else {
                    float dotx = 0, doty = 0;
                    dotx += vn->n[0] * mRsp->current_lookat_coeffs[0][0];
                    dotx += vn->n[1] * mRsp->current_lookat_coeffs[0][1];
                    dotx += vn->n[2] * mRsp->current_lookat_coeffs[0][2];
                    doty += vn->n[0] * mRsp->current_lookat_coeffs[1][0];
                    doty += vn->n[1] * mRsp->current_lookat_coeffs[1][1];
                    doty += vn->n[2] * mRsp->current_lookat_coeffs[1][2];

                    dotx /= 127.0f;
                    doty /= 127.0f;

                    dotx = Ship::Math::clamp(dotx, -1.0f, 1.0f);
                    doty = Ship::Math::clamp(doty, -1.0f, 1.0f);

                    if (mRsp->geometry_mode & G_TEXTURE_GEN_LINEAR) {
                        dotx = acosf(-dotx) /* M_PI */ * 0.159155f;
                        doty = acosf(-doty) /* M_PI */ * 0.159155f;
                    } else {
                        dotx = (dotx + 1.0f) / 4.0f;
                        doty = (doty + 1.0f) / 4.0f;
                    }

                    U = (int32_t)(dotx * mRsp->texture_scaling_factor.s);
                    V = (int32_t)(doty * mRsp->texture_scaling_factor.t);
                }
            }
        } else {
            d->color.r = v->cn[0];
            d->color.g = v->cn[1];
            d->color.b = v->cn[2];
        }

        d->u = U;
        d->v = V;

        // trivial clip rejection
        d->clip_rej = 0;
        if (x < -w) {
            d->clip_rej |= 1; // CLIP_LEFT
        }
        if (x > w) {
            d->clip_rej |= 2; // CLIP_RIGHT
        }
        if (y < -w) {
            d->clip_rej |= 4; // CLIP_BOTTOM
        }
        if (y > w) {
            d->clip_rej |= 8; // CLIP_TOP
        }
        // if (z < -w) d->clip_rej |= 16; // CLIP_NEAR
        if (z > w) {
            d->clip_rej |= 32; // CLIP_FAR
        }

        d->x = x;
        d->y = y;
        d->z = z;
        d->w = w;

        if (mRsp->geometry_mode & G_FOG) {
            /* The RSP computes fog from z/w, which hardware keeps inside [-1, 1] by clipping
               before fog is evaluated. This interpreter renders unclipped near/behind-camera
               vertices to match the reject microcodes, where z/w blows past that domain and
               F-Zero X's steep fog curve saturates whole surfaces. Clamp to the hardware domain;
               a vertex at or behind the camera takes the near (fog = 0) end. */
            float fogRatio;
            if (w <= 0.001f) {
                fogRatio = -1.0f;
            } else {
                fogRatio = Ship::Math::clamp(z / w, -1.0f, 1.0f);
            }

            float fog_z = fogRatio * mRsp->fog_mul + mRsp->fog_offset;
            fog_z = Ship::Math::clamp(fog_z, 0.0f, 255.0f);
            d->color.a = fog_z; // Use alpha variable to store fog factor

            // GDX_DIAG_FOG=1: log per-vertex fog inputs and result during a race.
            static const bool sDiagFog = std::getenv("GDX_DIAG_FOG") != nullptr;
            if (sDiagFog && gGdxRaceActive != 0) {
                static int sFogTraceCount = 0;
                if (sFogTraceCount < 2000) {
                    ++sFogTraceCount;
                    FILE* tf = fopen("fog-trace.txt", sFogTraceCount == 1 ? "w" : "a");
                    if (tf != nullptr) {
                        fprintf(tf, "F fm=%d fo=%d z=%.1f w=%.1f r=%.4f fog=%.1f\n", mRsp->fog_mul,
                                mRsp->fog_offset, z, w, fogRatio, fog_z);
                        fclose(tf);
                    }
                }
            }
        } else if (hasF3dflxAlpha) {
            d->color.a = f3dflxAlpha;
        } else {
            d->color.a = v->cn[3];
        }
    }
}

void Interpreter::GfxSpModifyVertex(uint16_t vtx_idx, uint8_t where, uint32_t val) {
    SUPPORT_CHECK(where == G_MWO_POINT_ST);

    int16_t s = (int16_t)(val >> 16);
    int16_t t = (int16_t)val;

    LoadedVertex* v = &mRsp->loaded_vertices[vtx_idx];
    v->u = s;
    v->v = t;
}

static inline uint32_t color_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d);
static inline uint32_t alpha_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d);

static LoadedVertex InterpolateLoadedVertex(const LoadedVertex& a, const LoadedVertex& b, float t) {
    LoadedVertex result{};
    result.x = a.x + (b.x - a.x) * t;
    result.y = a.y + (b.y - a.y) * t;
    result.z = a.z + (b.z - a.z) * t;
    result.w = a.w + (b.w - a.w) * t;
    result.u = a.u + (b.u - a.u) * t;
    result.v = a.v + (b.v - a.v) * t;
    result.color.r = static_cast<uint8_t>(std::clamp(a.color.r + (b.color.r - a.color.r) * t, 0.0f, 255.0f));
    result.color.g = static_cast<uint8_t>(std::clamp(a.color.g + (b.color.g - a.color.g) * t, 0.0f, 255.0f));
    result.color.b = static_cast<uint8_t>(std::clamp(a.color.b + (b.color.b - a.color.b) * t, 0.0f, 255.0f));
    result.color.a = static_cast<uint8_t>(std::clamp(a.color.a + (b.color.a - a.color.a) * t, 0.0f, 255.0f));
    return result;
}

static void UpdateLoadedVertexClipFlags(LoadedVertex& vertex) {
    vertex.clip_rej = 0;
    if (vertex.x < -vertex.w) {
        vertex.clip_rej |= 1;
    }
    if (vertex.x > vertex.w) {
        vertex.clip_rej |= 2;
    }
    if (vertex.y < -vertex.w) {
        vertex.clip_rej |= 4;
    }
    if (vertex.y > vertex.w) {
        vertex.clip_rej |= 8;
    }
    if (vertex.z < -vertex.w) {
        vertex.clip_rej |= 16;
    }
    if (vertex.z > vertex.w) {
        vertex.clip_rej |= 32;
    }
}

// GDX_RECT_HALF_TEXEL=1 extends the BILERP half-texel offset to texture rectangles; default off.
// It was briefly default-on and reverted: it did not fix the glyph defect it targeted (whose tile
// is CLAMP-addressed, where the supposed texel -1 bleed cannot occur), and rect quads already land
// pixel centres on texel centres, so the extra half texel blurs every BILERP rect in the game.
// Kept behind the switch so the hypothesis stays cheap to re-test.
static const bool sGdxRectHalfTexelDisabled = [] {
    const char* v = std::getenv("GDX_RECT_HALF_TEXEL");
    return !(v != nullptr && v[0] == '1');
}();

void Interpreter::GfxSpTri1(uint8_t vtx1_idx, uint8_t vtx2_idx, uint8_t vtx3_idx, bool is_rect) {
    if (mVrSuppressDraw && !mVrInHudSprites) {
        return;
    }
    if (mVrInHudSprites && mVrEyeActive && mVrPass == VrPass::WorldOnly) {
        return; // HUD-bracketed drawing (e.g. the FINISH letters) belongs to the HUD layer only
    }
    // A/B isolation: skip every triangle drawn with the combine mode given in
    // GDX_DIAG_SKIP_COMBINE (hex). Whatever vanishes from the frame identifies
    // the mesh that owns that material.
    static const uint64_t sSkipCombine = [] {
        const char* env = std::getenv("GDX_DIAG_SKIP_COMBINE");
        return (env != nullptr) ? strtoull(env, nullptr, 16) : 0ULL;
    }();
    if (sSkipCombine != 0 && !is_rect && mRdp->combine_mode == sSkipCombine) {
        return;
    }
    // Clipped polygons are triangulated by recursively submitting a fan through
    // this function. Do not clip those generated triangles a second time:
    // floating-point intersections can land infinitesimally outside a plane
    // and otherwise recurse until the stack overflows.
    static thread_local uint32_t sClipFanDepth = 0;
    struct LoadedVertex* v1 = &mRsp->loaded_vertices[vtx1_idx];
    struct LoadedVertex* v2 = &mRsp->loaded_vertices[vtx2_idx];
    struct LoadedVertex* v3 = &mRsp->loaded_vertices[vtx3_idx];
    struct LoadedVertex* v_arr[3] = { v1, v2, v3 };

    // if (rand()%2) return;

    if (!is_rect) {
        mGeometryDiagnostics.trianglesSubmitted++;
    }

    /* [fontmach] probe E: the Create Machine preview renders as uniform ENV color,
       implicating vertex SHADE == 0. Log one line per distinct lighting/shade input
       state so the zeroed source (light colors, ambient, or vertex color) is named. */
    if (!is_rect) {
        static const bool sDiagFontMachineE = std::getenv("GDX_DIAG_FONT_MACHINE") != nullptr;
        if (sDiagFontMachineE) {
            const bool lit = (mRsp->geometry_mode & G_LIGHTING) != 0;
            const int numLights = mRsp->current_num_lights;
            const int ambIdx = numLights > 0 ? numLights - 1 : 0;
            const F3DLight_t& amb = mRsp->current_lights[ambIdx].l;
            const F3DLight_t& l0 = mRsp->current_lights[0].l;
            /* Key on material identity, not per-vertex shade: keying on vertex color
               exhausts the ledger on one mesh's gradient. */
            const uint64_t shadeKey = (static_cast<uint64_t>(lit) << 63) |
                                      (static_cast<uint64_t>(numLights & 0xF) << 58) |
                                      (static_cast<uint64_t>(amb.col[0] & 0xF8) << 47) |
                                      (static_cast<uint64_t>(amb.col[2] & 0xF8) << 42) |
                                      (static_cast<uint64_t>(l0.col[0] & 0xF8) << 37) |
                                      (static_cast<uint64_t>(l0.col[2] & 0xF8) << 32) |
                                      (static_cast<uint64_t>(mRdp->env_color.r) << 24) |
                                      (static_cast<uint64_t>(mRdp->env_color.g) << 16) |
                                      (static_cast<uint64_t>(mRdp->env_color.b) << 8) |
                                      ((mRdp->combine_mode ^ (mRdp->combine_mode >> 32)) & 0xFF);
            static uint64_t sSeenShadeStates[64] = {};
            static int sSeenShadeCount = 0;
            bool seenShade = false;
            for (int s = 0; s < sSeenShadeCount; s++) {
                if (sSeenShadeStates[s] == shadeKey) {
                    seenShade = true;
                    break;
                }
            }
            if (!seenShade && sSeenShadeCount < 64) {
                sSeenShadeStates[sSeenShadeCount++] = shadeKey;
                /* Sum the first 64 source bytes at draw time: zero convicts slot
                   replacement between load and draw, nonzero points at UV/handle. */
                const int probeTile = mRdp->first_tile_index;
                const uint32_t probeTmem = mRdp->texture_tile[probeTile].tmem_index;
                const uint8_t* probeSrc = mRdp->loaded_texture[probeTmem].addr;
                const uint32_t probeSize = mRdp->loaded_texture[probeTmem].size_bytes;
                uint32_t srcSum = 0;
                if (probeSrc != nullptr) {
                    const uint32_t n = probeSize < 64 ? probeSize : 64;
                    for (uint32_t k = 0; k < n; k++) {
                        srcSum = srcSum * 31 + probeSrc[k];
                    }
                }
                gdx_dbg_logf("[fontmach] shade-src tile=%d tmem=0x%X addr=%p sizeB=%u sum=%08X\n",
                             probeTile, probeTmem, static_cast<const void*>(probeSrc), probeSize, srcSum);
                gdx_dbg_logf("[fontmach] shade lit=%d n=%d v0=(%u,%u,%u,%u) amb=(%u,%u,%u) "
                             "l0=(%u,%u,%u dir %d,%d,%d) env=(%u,%u,%u,%u) cc=%016llX "
                             "tile=%u st0=(%.1f,%.1f) st1=(%.1f,%.1f) st2=(%.1f,%.1f)\n",
                             lit ? 1 : 0, numLights, v1->color.r, v1->color.g, v1->color.b, v1->color.a,
                             amb.col[0], amb.col[1], amb.col[2], l0.col[0], l0.col[1], l0.col[2],
                             (int)l0.dir[0], (int)l0.dir[1], (int)l0.dir[2], mRdp->env_color.r,
                             mRdp->env_color.g, mRdp->env_color.b, mRdp->env_color.a,
                             (unsigned long long)mRdp->combine_mode, (unsigned)mRdp->first_tile_index,
                             v1->u, v1->v, v2->u, v2->v, v3->u, v3->v);
            }
        }
    }

    // The F3DEX2.Rej family does not clip: it drops any triangle with a vertex outside the reject
    // box (twice the viewport extent in X/Y) or behind the eye. F-Zero X relies on this for the
    // machine screens and FLX vehicles; rasterizing those triangles paints giant wedges.
    if (!is_rect && sClipFanDepth == 0 &&
        (mF3dex2Variant == F3dex2Variant::Reject || mF3dex2Variant == F3dex2Variant::FZeroFlxReject)) {
        for (const LoadedVertex* vertex : v_arr) {
            const float w2 = 2.0f * vertex->w;
            if (vertex->w <= 0.0f || vertex->x < -w2 || vertex->x > w2 || vertex->y < -w2 ||
                vertex->y > w2 || vertex->z < -vertex->w) {
                mGeometryDiagnostics.trianglesClipRejected++;
                return;
            }
        }
    }

    // Clip in homogeneous space before culling or perspective division. F3DEX/F3DFLX display
    // lists routinely straddle the eye and frustum planes, and a modern backend renders those as
    // giant wedges instead of the RSP-clipped polygons.
    constexpr float kMinClipW = 1.0e-5f;
    const auto clipDistance = [=](const LoadedVertex& vertex, uint32_t plane) {
        switch (plane) {
            case 0:
                return vertex.w - kMinClipW;
            case 1:
                return vertex.x + vertex.w;
            case 2:
                return vertex.w - vertex.x;
            case 3:
                return vertex.y + vertex.w;
            case 4:
                return vertex.w - vertex.y;
            case 5:
                return vertex.z + vertex.w;
            default:
                return vertex.w - vertex.z;
        }
    };
    bool requiresClipping = false;
    if (!is_rect && sClipFanDepth == 0) {
        for (const LoadedVertex* vertex : v_arr) {
            for (uint32_t plane = 0; plane < 7; ++plane) {
                if (clipDistance(*vertex, plane) < 0.0f) {
                    requiresClipping = true;
                    break;
                }
            }
            if (requiresClipping) {
                break;
            }
        }
    }

    if (requiresClipping) {
        LoadedVertex polygonA[16] = { *v1, *v2, *v3 };
        LoadedVertex polygonB[16]{};
        LoadedVertex* input = polygonA;
        LoadedVertex* output = polygonB;
        size_t inputCount = 3;

        for (uint32_t plane = 0; plane < 7 && inputCount >= 3; ++plane) {
            size_t outputCount = 0;
            for (size_t i = 0; i < inputCount; ++i) {
                const LoadedVertex& current = input[i];
                const LoadedVertex& next = input[(i + 1) % inputCount];
                const float currentDistance = clipDistance(current, plane);
                const float nextDistance = clipDistance(next, plane);
                const bool currentInside = currentDistance >= 0.0f;
                const bool nextInside = nextDistance >= 0.0f;

                if (currentInside && outputCount < 16) {
                    output[outputCount++] = current;
                }
                if (currentInside != nextInside && outputCount < 16) {
                    const float denominator = currentDistance - nextDistance;
                    const float t = denominator != 0.0f ? currentDistance / denominator : 0.0f;
                    output[outputCount++] = InterpolateLoadedVertex(current, next, std::clamp(t, 0.0f, 1.0f));
                }
            }
            inputCount = outputCount;
            std::swap(input, output);
        }

        if (inputCount < 3) {
            mGeometryDiagnostics.trianglesClipRejected++;
            return;
        }

        LoadedVertex saved[3];
        for (size_t i = 0; i < 3; ++i) {
            saved[i] = mRsp->loaded_vertices[MAX_VERTICES + i];
        }

        // Three reserved vertex slots suffice regardless of polygon size.
        mGeometryDiagnostics.trianglesSubmitted--;
        ++sClipFanDepth;
        for (size_t i = 1; i + 1 < inputCount; ++i) {
            mRsp->loaded_vertices[MAX_VERTICES] = input[0];
            mRsp->loaded_vertices[MAX_VERTICES + 1] = input[i];
            mRsp->loaded_vertices[MAX_VERTICES + 2] = input[i + 1];
            for (size_t j = 0; j < 3; ++j) {
                mRsp->loaded_vertices[MAX_VERTICES + j].w =
                    std::max(mRsp->loaded_vertices[MAX_VERTICES + j].w, kMinClipW);
                UpdateLoadedVertexClipFlags(mRsp->loaded_vertices[MAX_VERTICES + j]);
            }
            GfxSpTri1(MAX_VERTICES, MAX_VERTICES + 1, MAX_VERTICES + 2, false);
        }
        --sClipFanDepth;

        for (size_t i = 0; i < 3; ++i) {
            mRsp->loaded_vertices[MAX_VERTICES + i] = saved[i];
        }
        return;
    }

    if (v1->clip_rej & v2->clip_rej & v3->clip_rej) {
        // The whole triangle lies outside the visible area
        if (!is_rect) {
            mGeometryDiagnostics.trianglesClipRejected++;
        }
        return;
    }

    const uint32_t cull_both = get_attr(CULL_BOTH);
    const uint32_t cull_front = get_attr(CULL_FRONT);
    const uint32_t cull_back = get_attr(CULL_BACK);

    if ((mRsp->geometry_mode & cull_both) != 0) {
        float dx1 = v1->x / (v1->w) - v2->x / (v2->w);
        float dy1 = v1->y / (v1->w) - v2->y / (v2->w);
        float dx2 = v3->x / (v3->w) - v2->x / (v2->w);
        float dy2 = v3->y / (v3->w) - v2->y / (v2->w);
        float cross = dx1 * dy2 - dy1 * dx2;

        if ((v1->w < 0) ^ (v2->w < 0) ^ (v3->w < 0)) {
            // If one vertex lies behind the eye, negating cross will give the correct result.
            // If all vertices lie behind the eye, the triangle will be rejected anyway.
            cross = -cross;
        }

        // G_EX_INVERT_CULLING is a LUS extension, not tied to a specific ucode,
        // so apply it regardless of the active microcode handler.
        if ((mRsp->extra_geometry_mode & G_EX_INVERT_CULLING) != 0) {
            cross = -cross;
        }

        auto cull_type = mRsp->geometry_mode & cull_both;

        if (cull_type == cull_front) {
            if (cross <= 0) {
                if (!is_rect) {
                    mGeometryDiagnostics.trianglesCullRejected++;
                }
                return;
            }
        } else if (cull_type == cull_back) {
            if (cross >= 0) {
                if (!is_rect) {
                    mGeometryDiagnostics.trianglesCullRejected++;
                }
                return;
            }
        } else if (cull_type == cull_both) {
            // Why is this even an option?
            if (!is_rect) {
                mGeometryDiagnostics.trianglesCullRejected++;
            }
            return;
        }
    }

    // Capture the first oversized triangle that survives Reject-variant screening each frame.
    // Legit machine-select geometry stays under ~0.2 NDC extent, so anything near the reject-box
    // limit is a defect candidate worth identifying by material.
    if (!is_rect &&
        (mF3dex2Variant == F3dex2Variant::Reject || mF3dex2Variant == F3dex2Variant::FZeroFlxReject)) {
        float extent = 0.0f;
        for (const LoadedVertex* vertex : v_arr) {
            if (vertex->w > 1.0e-6f) {
                extent = std::max(extent, std::max(fabsf(vertex->x), fabsf(vertex->y)) / vertex->w);
            }
        }
        if (extent > 0.9f) {
            if (mGeometryDiagnostics.bigTriangles++ == 0) {
                for (int i = 0; i < 3; i++) {
                    mGeometryDiagnostics.bigTriX[i] = v_arr[i]->x;
                    mGeometryDiagnostics.bigTriY[i] = v_arr[i]->y;
                    mGeometryDiagnostics.bigTriZ[i] = v_arr[i]->z;
                    mGeometryDiagnostics.bigTriW[i] = v_arr[i]->w;
                }
                mGeometryDiagnostics.bigTriGeometryMode = mRsp->geometry_mode;
                mGeometryDiagnostics.bigTriCombine = mRdp->combine_mode;
                const uint32_t bigTriTile = mRdp->first_tile_index;
                mGeometryDiagnostics.bigTriTile = static_cast<uint8_t>(bigTriTile);
                mGeometryDiagnostics.bigTriTexture =
                    mRdp->loaded_texture[mRdp->texture_tile[bigTriTile].tmem_index].addr;
                mGeometryDiagnostics.bigTriViewportX = mRdp->viewport.x;
                mGeometryDiagnostics.bigTriViewportY = mRdp->viewport.y;
                mGeometryDiagnostics.bigTriViewportW = mRdp->viewport.width;
                mGeometryDiagnostics.bigTriViewportH = mRdp->viewport.height;
            }
        }
    }

    // depth_test is set when the fragment has a depth value to compare (either from vertex Z via
    // RSP G_ZBUFFER, or from the prim-depth register via G_ZS_PRIM) and Z_CMP is requested.
    bool zbuffer_enabled = (mRsp->geometry_mode & G_ZBUFFER) == G_ZBUFFER;
    bool prim_depth_enabled = (mRdp->other_mode_l & G_ZS_PRIM) != 0;
    bool depth_test = (zbuffer_enabled || prim_depth_enabled) && (mRdp->other_mode_l & Z_CMP) == Z_CMP;
    bool depth_mask = (mRdp->other_mode_l & Z_UPD) == Z_UPD;
    static const bool diagnosticDisablePreFlxDepth = std::getenv("GDX_DIAG_DISABLE_PREFLX_DEPTH") != nullptr;
    if (diagnosticDisablePreFlxDepth && mF3dex2Variant != F3dex2Variant::FZeroFlxReject && depth_test) {
        depth_test = false;
        depth_mask = false;
        mGeometryDiagnostics.depthBypassTriangles++;
    }
    uint8_t depth_test_and_mask = (depth_test ? 1 : 0) | (depth_mask ? 2 : 0);
    if (depth_test_and_mask != mRenderingState.depth_test_and_mask) {
        Flush();
        mRapi->SetDepthTestAndMask(depth_test, depth_mask);
        mRenderingState.depth_test_and_mask = depth_test_and_mask;
    }

    bool zmode_decal = (mRdp->other_mode_l & ZMODE_DEC) == ZMODE_DEC;
    if (zmode_decal != mRenderingState.decal_mode) {
        Flush();
        mRapi->SetZmodeDecal(zmode_decal);
        mRenderingState.decal_mode = zmode_decal;
    }

    if (mRdp->viewport_or_scissor_changed) {
        if (memcmp(&mRdp->viewport, &mRenderingState.viewport, sizeof(mRdp->viewport)) != 0) {
            Flush();
            mRapi->SetViewport(mRdp->viewport.x, mRdp->viewport.y, mRdp->viewport.width, mRdp->viewport.height);
            mRenderingState.viewport = mRdp->viewport;
        }
        if (memcmp(&mRdp->scissor, &mRenderingState.scissor, sizeof(mRdp->scissor)) != 0) {
            Flush();
            mRapi->SetScissor(mRdp->scissor.x, mRdp->scissor.y, mRdp->scissor.width, mRdp->scissor.height);
            mRenderingState.scissor = mRdp->scissor;
        }
        mRdp->viewport_or_scissor_changed = false;
    }

    uint64_t cc_id = mRdp->combine_mode;
    uint64_t cc_options = 0;
    bool use_alpha = ((mRdp->other_mode_l & (3 << 20)) == (G_BL_CLR_MEM << 20) &&
                      (mRdp->other_mode_l & (3 << 16)) == (G_BL_1MA << 16)) ||
                     ((mRdp->other_mode_l & (3 << 22)) == (G_BL_CLR_MEM << 22) &&
                      (mRdp->other_mode_l & (3 << 18)) == (G_BL_1MA << 18));
    uint8_t blend_src = mRdp->other_mode_l >> 30;
    bool use_blend_color = blend_src == G_BL_CLR_BL;
    constexpr uint32_t constantBlendMode =
        (GBL_c1(G_BL_CLR_BL, G_BL_0, G_BL_CLR_BL, G_BL_1)) |
        (GBL_c2(G_BL_CLR_BL, G_BL_0, G_BL_CLR_BL, G_BL_1));
    // Both cycles select only blend color; inherited material and fog alpha cannot contribute.
    const float blendFactor = (mRdp->other_mode_l & 0xFFFF0000u) == constantBlendMode
                                 ? 1.0f : mRdp->fog_color.a / 255.0f;
    bool use_fog = blend_src == G_BL_CLR_FOG || use_blend_color;
    const bool originalUseFog = use_fog;
    bool texture_edge = (mRdp->other_mode_l & CVG_X_ALPHA) == CVG_X_ALPHA;
    bool use_noise = (mRdp->other_mode_l & (3U << G_MDSFT_ALPHACOMPARE)) == G_AC_DITHER;
    bool use_2cyc = (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE)) == G_CYC_2CYCLE;
    bool alpha_threshold = (mRdp->other_mode_l & (3U << G_MDSFT_ALPHACOMPARE)) == G_AC_THRESHOLD;
    bool invisible =
        (mRdp->other_mode_l & (3 << 24)) == (G_BL_0 << 24) && (mRdp->other_mode_l & (3 << 20)) == (G_BL_CLR_MEM << 20);
    if (invisible && !is_rect) {
        mGeometryDiagnostics.trianglesInvisible++;
    }
    bool use_grayscale = mRdp->grayscale;
    bool use_prim_depth = (mRdp->other_mode_l & G_ZS_PRIM) != 0;
    static const bool diagnosticDisablePreFlxFog =
        std::getenv("GDX_DIAG_DISABLE_PREFLX_FOG") != nullptr;
    const bool bypassFog =
        diagnosticDisablePreFlxFog && mF3dex2Variant != F3dex2Variant::FZeroFlxReject && originalUseFog && !is_rect;
    if (bypassFog) {
        use_fog = false;
    }
    // Create Machine flat-navy preview: skipping the gDPSetRenderMode call at
    // machine_create_draw.c:655 fixes it, but the mechanism is unproven. Log one line per distinct
    // mode tuple to diff against the machine-settings screen, which runs the same display lists
    // correctly. getenv rather than gdx_dev_gates, which is not visible from this TU.
    static const bool diagnosticBlendMode = std::getenv("GDX_DIAG_BLENDMODE") != nullptr;
    if (diagnosticBlendMode) {
        static std::set<std::pair<uint64_t, uint64_t>> sSeenModes;
        static int sModeLogs = 0;
        const auto key = std::make_pair(static_cast<uint64_t>(mRdp->other_mode_l), cc_id);

        if (sModeLogs < 48 && sSeenModes.insert(key).second) {
            ++sModeLogs;
            // cc_id packs colour in the low 16 bits and alpha above it.
            gdx_dbg_logf("[blendmode] other_l=%08X cc=%016llX ccAlpha=%04X use_alpha=%d decal=%d "
                         "forceBl=%d zmode=%X\n",
                         (unsigned) mRdp->other_mode_l, (unsigned long long) cc_id,
                         (unsigned) ((cc_id >> 16) & 0xFFFF), use_alpha ? 1 : 0,
                         ((mRdp->other_mode_l & ZMODE_DEC) == ZMODE_DEC) ? 1 : 0,
                         (mRdp->other_mode_l & FORCE_BL) ? 1 : 0,
                         (unsigned) ((mRdp->other_mode_l >> 10) & 3));
        }
    }

    static const bool diagnosticForcePreFlxSimpleMaterial =
        std::getenv("GDX_DIAG_FORCE_PREFLX_SIMPLE_MATERIAL") != nullptr;
    const bool forceSimpleMaterial =
        diagnosticForcePreFlxSimpleMaterial && mF3dex2Variant != F3dex2Variant::FZeroFlxReject && !is_rect;

    if (forceSimpleMaterial) {
        // A/B diagnostic: bypass original combiner/blender interpretation while
        // preserving geometry, texture state, UVs, and vertex shade color.
        cc_id = color_comb(G_CCMUX_TEXEL0, G_CCMUX_0, G_CCMUX_SHADE, G_CCMUX_0) |
                (static_cast<uint64_t>(
                     alpha_comb(G_ACMUX_TEXEL0, G_ACMUX_0, G_ACMUX_SHADE, G_ACMUX_0))
                 << 16);
        use_alpha = false;
        use_fog = false;
        texture_edge = false;
        use_noise = false;
        use_2cyc = false;
        alpha_threshold = false;
        invisible = false;
        use_grayscale = false;
        use_prim_depth = false;
    }

    if (texture_edge) {
        if (use_alpha) {
            alpha_threshold = true;
            texture_edge = false;
        }
        use_alpha = true;
    }

    // Alpha compare (G_AC_THRESHOLD / G_AC_DITHER) rejects fragments by texel alpha even when the
    // blender itself is opaque -- COPY-mode HUD sprite blits rely on it. The shader only compiles
    // the discard under the ALPHA option, so alpha compare must force it on, like texture_edge.
    if (alpha_threshold || use_noise) {
        use_alpha = true;
    }

    if (use_alpha) {
        cc_options |= SHADER_OPT(ALPHA);
    }
    if (use_fog) {
        cc_options |= SHADER_OPT(FOG);
    }
    if (texture_edge) {
        cc_options |= SHADER_OPT(TEXTURE_EDGE);
    }
    if (use_noise) {
        cc_options |= SHADER_OPT(NOISE);
    }
    if (use_2cyc) {
        cc_options |= SHADER_OPT(_2CYC);
    }
    if (alpha_threshold) {
        cc_options |= SHADER_OPT(ALPHA_THRESHOLD);
    }
    if (invisible) {
        cc_options |= SHADER_OPT(INVISIBLE);
    }
    if (use_grayscale) {
        cc_options |= SHADER_OPT(GRAYSCALE);
    }
    if (use_prim_depth) {
        cc_options |= SHADER_OPT(PRIM_DEPTH);
    }

    if (!mShaderStack.empty()) {
        cc_options |= (mShaderStack.top() << SHADER_ID_SHIFT);
    } else {
        cc_options |= -1 << SHADER_ID_SHIFT;
    }

    const uint32_t texel0Tile = mRdp->first_tile_index;
    // In two-cycle mode TEXEL1 uses the next tile descriptor. F-Zero X relies on this for track
    // and vehicle materials, where adjacent tiles describe different views of one TMEM load.
    // GDX_DIAG_TEXEL1_FROM_BASE forces TEXEL1 back to the base tile, to isolate whether
    // track-floor corruption comes from that adjacent-tile view.
    static const bool diagnosticTexel1FromBase = std::getenv("GDX_DIAG_TEXEL1_FROM_BASE") != nullptr;
    const uint32_t texel1Tile =
        diagnosticTexel1FromBase ? texel0Tile : ((mRdp->first_tile_index + 1u) & 7u);
    const auto& texel0Loaded = mRdp->loaded_texture[mRdp->texture_tile[texel0Tile].tmem_index];
    const auto& texel1Loaded = mRdp->loaded_texture[mRdp->texture_tile[texel1Tile].tmem_index];

    if (texel0Loaded.masked) {
        cc_options |= SHADER_OPT(TEXEL0_MASK);
    }
    if (texel1Loaded.masked) {
        cc_options |= SHADER_OPT(TEXEL1_MASK);
    }
    if (texel0Loaded.blended) {
        cc_options |= SHADER_OPT(TEXEL0_BLEND);
    }
    if (texel1Loaded.blended) {
        cc_options |= SHADER_OPT(TEXEL1_BLEND);
    }

    ColorCombinerKey key;
    key.combine_mode = cc_id;
    key.options = cc_options;

    ColorCombiner* comb = LookupOrCreateColorCombiner(key);

    // GDX_DIAG_EFFECTDRAW: the PIT/HEAL strip draws flat tan despite a proven TMEM identity. Log
    // the combine id and the texel slots the combiner uses, to confirm or rule out two-cycle
    // TEXEL1 sampling tile+1, which for PIT is DIRT's I4/IA4 mask.
    static const bool sDiagEffectDraw = std::getenv("GDX_DIAG_EFFECTDRAW") != nullptr;
    if (sDiagEffectDraw && gGdxRaceActive != 0 && texel0Tile >= 1 && texel0Tile <= 4) {
        static int sEffectDrawStateLogs = 0;
        if (sEffectDrawStateLogs < 64) {
            ++sEffectDrawStateLogs;
            SPDLOG_ERROR("[effect-draw-state] tile0={} tile1={} cc_id={:#x} usedTex0={} usedTex1={} "
                         "tile0_fmt={} tile0_siz={} tile0_cms={} tile0_cmt={} tile1_fmt={} tile1_siz={} "
                         "tile1_cms={} tile1_cmt={} 2cyc={}",
                         texel0Tile, texel1Tile, cc_id, comb->usedTextures[0], comb->usedTextures[1],
                         mRdp->texture_tile[texel0Tile].fmt, mRdp->texture_tile[texel0Tile].siz,
                         mRdp->texture_tile[texel0Tile].cms, mRdp->texture_tile[texel0Tile].cmt,
                         mRdp->texture_tile[texel1Tile].fmt, mRdp->texture_tile[texel1Tile].siz,
                         mRdp->texture_tile[texel1Tile].cms, mRdp->texture_tile[texel1Tile].cmt, use_2cyc);
        }
    }

    // GDX_MINIMAP_PROBE=1: confirm the texture filter actually in effect and the
    // resolved combiner for the CI8 minimap rects. Pairs with [minimap-ci8] above.
    static const bool sMinimapDrawProbe = std::getenv("GDX_MINIMAP_PROBE") != nullptr;
    if (sMinimapDrawProbe && is_rect &&
        mRdp->texture_tile[texel0Tile].fmt == G_IM_FMT_CI &&
        mRdp->texture_tile[texel0Tile].siz == G_IM_SIZ_8b) {
        static int sMinimapDrawLogs = 0;
        if (sMinimapDrawLogs < 32) {
            ++sMinimapDrawLogs;
            uint32_t textfilt = (mRdp->other_mode_h >> G_MDSFT_TEXTFILT) & 3u;
            uint32_t tile_w = (uint32_t)((mRdp->texture_tile[texel0Tile].lrs -
                                          mRdp->texture_tile[texel0Tile].uls + 4) / 4);
            uint32_t tile_h = (uint32_t)((mRdp->texture_tile[texel0Tile].lrt -
                                          mRdp->texture_tile[texel0Tile].ult + 4) / 4);
            SPDLOG_ERROR("[minimap-draw] is_rect cc_id={:#x} usedTex0={} usedTex1={} "
                         "textfilt={} (0=POINT 2=BILERP 3=AVERAGE) fmt={} siz={} cms={} cmt={} tile={}x{}",
                         cc_id, comb->usedTextures[0], comb->usedTextures[1], textfilt,
                         mRdp->texture_tile[texel0Tile].fmt, mRdp->texture_tile[texel0Tile].siz,
                         mRdp->texture_tile[texel0Tile].cms, mRdp->texture_tile[texel0Tile].cmt,
                         tile_w, tile_h);
        }
    }

    uint32_t tm = 0;
    uint32_t tex_width[2], tex_height[2], tex_width2[2], tex_height2[2];
    uint32_t effective_tile[2];

    for (int i = 0; i < 2; i++) {
        uint32_t tile = (i == 1) ? texel1Tile : texel0Tile;
        effective_tile[i] = tile;

        if (comb->usedTextures[i]) {
            const auto& activeLoaded = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index];
            if (mRdp->textures_changed[i]) {
                Flush();
                ImportTexture(i, tile, false);
                if (activeLoaded.masked) {
                    ImportTextureMask(SHADER_FIRST_MASK_TEXTURE + i, tile);
                }
                if (activeLoaded.blended) {
                    ImportTexture(SHADER_FIRST_REPLACEMENT_TEXTURE + i, tile, true);
                }
                mRdp->textures_changed[i] = false;
            }

            uint8_t cms = mRdp->texture_tile[tile].cms;
            uint8_t cmt = mRdp->texture_tile[tile].cmt;

            uint32_t loaded_line_size = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].line_size_bytes;
            uint32_t loaded_size = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].size_bytes;
            uint32_t loaded_full_line =
                mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].full_image_line_size_bytes;
            uint32_t tex_size_bytes;
            uint32_t line_size;
            if ((loaded_line_size != loaded_size || loaded_full_line != loaded_size) && loaded_line_size > 0) {
                line_size = loaded_line_size;
                tex_size_bytes = loaded_size;
            } else {
                line_size = mRdp->texture_tile[tile].line_size_bytes;
                tex_size_bytes = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index].orig_size_bytes;
                // RGBA32: texture_tile stores TMEM-interleaved stride (half of actual DRAM stride).
                if (mRdp->texture_tile[tile].siz == G_IM_SIZ_32b) {
                    line_size *= 2;
                }
            }

            if (line_size == 0) {
                line_size = 1;
            }

            tex_height[i] = tex_size_bytes / line_size;
            switch (mRdp->texture_tile[tile].siz) {
                case G_IM_SIZ_4b:
                    line_size <<= 1;
                    break;
                case G_IM_SIZ_8b:
                    break;
                case G_IM_SIZ_16b:
                    line_size /= G_IM_SIZ_16b_LINE_BYTES;
                    break;
                case G_IM_SIZ_32b:
                    line_size /= 4; // RGBA32: 4 bytes per pixel (line_size is now actual DRAM stride)
                    break;
            }
            tex_width[i] = line_size;

            tex_width2[i] = (uint32_t)(int32_t)((mRdp->texture_tile[tile].lrs - mRdp->texture_tile[tile].uls + 4) / 4);
            tex_height2[i] = (uint32_t)(int32_t)((mRdp->texture_tile[tile].lrt - mRdp->texture_tile[tile].ult + 4) / 4);

            // Same pyramid-like ratio gate as ImportTexture: only clamp when loaded pixels
            // are close to rendered pixels (mipmap), not when much bigger (window scroll).
            uint32_t loadedPx = tex_width[i] * tex_height[i];
            uint32_t renderedPx = tex_width2[i] * tex_height2[i];
            bool pyrLike = renderedPx > 0 && loadedPx > renderedPx && loadedPx * 8 < renderedPx * 13;
            if ((pyrLike || (cms & G_TX_CLAMP)) && tex_width2[i] > 0 && tex_width2[i] < tex_width[i]) {
                tex_width[i] = tex_width2[i];
            }
            if ((pyrLike || (cmt & G_TX_CLAMP)) && tex_height2[i] > 0 && tex_height2[i] < tex_height[i]) {
                tex_height[i] = tex_height2[i];
            }
            ApplyTileMaskExtent(mRdp, tile, tex_width[i], tex_height[i], /*maskAuthoritative=*/true);
            if (IsScaledLinearTexture(activeLoaded.raw_tex_metadata) &&
                !ScaledTextureExtent(mRdp, tile, mRapi->GetMaxTextureSize(), tex_width[i], tex_height[i])) {
                return;
            }
            /* One line per distinct draw-time UV-extent state: tex_w/h feed texcoord
               normalization, and a mismatch against the uploaded decode extent is the bug. */
            {
                static const bool sDiagFontMachineD = std::getenv("GDX_DIAG_FONT_MACHINE") != nullptr;
                if (sDiagFontMachineD) {
                    const uint64_t drawKey =
                        (static_cast<uint64_t>(tile) << 56) |
                        (static_cast<uint64_t>(mRdp->texture_tile[tile].tmem_index) << 40) |
                        (static_cast<uint64_t>(mRdp->texture_tile[tile].fmt) << 37) |
                        (static_cast<uint64_t>(mRdp->texture_tile[tile].siz) << 35) |
                        (static_cast<uint64_t>(tex_width[i] & 0x7FF) << 24) |
                        (static_cast<uint64_t>(tex_height[i] & 0x7FF) << 13) |
                        (static_cast<uint64_t>(tex_width2[i] & 0x3F) << 7) | (tex_height2[i] & 0x7F);
                    static uint64_t sSeenDraws[128] = {};
                    static int sSeenDrawCount = 0;
                    bool seenDraw = false;
                    for (int s = 0; s < sSeenDrawCount; s++) {
                        if (sSeenDraws[s] == drawKey) {
                            seenDraw = true;
                            break;
                        }
                    }
                    if (!seenDraw && sSeenDrawCount < 128) {
                        sSeenDraws[sSeenDrawCount++] = drawKey;
                        gdx_dbg_logf("[fontmach] draw tile=%u tmem=0x%X fmt=%u siz=%u texWH=%ux%u "
                                     "tileWH=%ux%u mask=%u/%u shift=%u/%u cm=%u/%u rect=%d scaleST=%X/%X\n",
                                     tile, mRdp->texture_tile[tile].tmem_index, mRdp->texture_tile[tile].fmt,
                                     mRdp->texture_tile[tile].siz, tex_width[i], tex_height[i], tex_width2[i],
                                     tex_height2[i], mRdp->texture_tile[tile].masks, mRdp->texture_tile[tile].maskt,
                                     mRdp->texture_tile[tile].shifts, mRdp->texture_tile[tile].shiftt,
                                     mRdp->texture_tile[tile].cms, mRdp->texture_tile[tile].cmt, is_rect ? 1 : 0,
                                     mRsp->texture_scaling_factor.s, mRsp->texture_scaling_factor.t);
                    }
                }
            }
            // Degenerate load bookkeeping (a slot whose recorded byte count is smaller than one
            // line) yields a zero extent, and the texcoord normalization below would divide by
            // zero and emit infinite UVs. The tile mask is the hardware wrap period, so prefer it;
            // otherwise clamp to a single texel.
            if (tex_width[i] == 0) {
                const uint8_t masksFix = mRdp->texture_tile[tile].masks;
                tex_width[i] = (masksFix != G_TX_NOMASK && masksFix < 31) ? (1u << masksFix) : 1u;
            }
            if (tex_height[i] == 0) {
                const uint8_t masktFix = mRdp->texture_tile[tile].maskt;
                tex_height[i] = (masktFix != G_TX_NOMASK && masktFix < 31) ? (1u << masktFix) : 1u;
            }
            if (!is_rect && i == 0) {
                mGeometryDiagnostics.textureWidth = tex_width[i];
                mGeometryDiagnostics.textureHeight = tex_height[i];
                mGeometryDiagnostics.textureLineBytes = loaded_line_size;
                mGeometryDiagnostics.textureSizeBytes = loaded_size;
                mGeometryDiagnostics.textureTmem = mRdp->texture_tile[tile].tmem_index;
                mGeometryDiagnostics.textureTile = static_cast<uint8_t>(tile);
                mGeometryDiagnostics.textureMaskS = mRdp->texture_tile[tile].masks;
                mGeometryDiagnostics.textureMaskT = mRdp->texture_tile[tile].maskt;
                mGeometryDiagnostics.textureScaleS = mRsp->texture_scaling_factor.s;
                mGeometryDiagnostics.textureScaleT = mRsp->texture_scaling_factor.t;
            }

            uint32_t tex_width1 = tex_width[i] << (cms & G_TX_MIRROR);
            uint32_t tex_height1 = tex_height[i] << (cmt & G_TX_MIRROR);

            if ((cms & G_TX_CLAMP) && ((cms & G_TX_MIRROR) || tex_width1 != tex_width2[i])) {
                tm |= 1 << 2 * i;
                cms &= ~G_TX_CLAMP;
            }
            if ((cmt & G_TX_CLAMP) && ((cmt & G_TX_MIRROR) || tex_height1 != tex_height2[i])) {
                tm |= 1 << 2 * i + 1;
                cmt &= ~G_TX_CLAMP;
            }

            if (mRenderingState.mTextures[i] == nullptr) {
                /* Log draws that skip sampler setup because the GPU texture handle is null
                   despite TMEM holding real bytes. */
                static const bool sDiagFontMachineA = std::getenv("GDX_DIAG_FONT_MACHINE") != nullptr;
                if (sDiagFontMachineA) {
                    static int sNullTexLogs = 0;
                    if (sNullTexLogs < 32) {
                        ++sNullTexLogs;
                        gdx_dbg_logf("[fontmach] NULL mTextures[%d] tile=%u tmem=0x%X fmt=%u siz=%u\n",
                                      i, (unsigned)(i == 0 ? mRdp->first_tile_index
                                                           : ((mRdp->first_tile_index + 1) & 7)),
                                      mRdp->texture_tile[i == 0 ? mRdp->first_tile_index
                                                                : ((mRdp->first_tile_index + 1) & 7)].tmem_index,
                                      mRdp->texture_tile[i == 0 ? mRdp->first_tile_index
                                                                : ((mRdp->first_tile_index + 1) & 7)].fmt,
                                      mRdp->texture_tile[i == 0 ? mRdp->first_tile_index
                                                                : ((mRdp->first_tile_index + 1) & 7)].siz);
                    }
                }
                continue;
            }

            // N64 G_TF_AVERAGE is defined only for aligned 1:1 copies and samples exact texels there;
            // treating it as linear smears 1px HUD detail (e.g. minimap border) at upscaled resolutions.
            bool linear_filter = (mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) == G_TF_BILERP;
            if (!mRenderingState.sampler_valid[i] ||
                linear_filter != mRenderingState.sampler_linear_filter[i] ||
                cms != mRenderingState.sampler_cms[i] || cmt != mRenderingState.sampler_cmt[i]) {
                Flush();

                // Set the same sampler params on the blended texture. Needed for opengl.
                if (activeLoaded.blended) {
                    mRapi->SetSamplerParameters(SHADER_FIRST_REPLACEMENT_TEXTURE + i, linear_filter, cms, cmt);
                }

                mRapi->SetSamplerParameters(i, linear_filter, cms, cmt);
                mRenderingState.sampler_valid[i] = true;
                mRenderingState.sampler_linear_filter[i] = linear_filter;
                mRenderingState.sampler_cms[i] = cms;
                mRenderingState.sampler_cmt[i] = cmt;
            }
        }
    }

    if (!is_rect) {
        mGeometryDiagnostics.lastShaderId0 = comb->shader_id0;
        mGeometryDiagnostics.lastShaderId1 = comb->shader_id1;
        if (originalUseFog) {
            mGeometryDiagnostics.fogTriangles++;
            for (const LoadedVertex* vertex : v_arr) {
                const float fogFactor =
                    use_blend_color ? blendFactor : vertex->color.a / 255.0f;
                mGeometryDiagnostics.minFogFactor =
                    std::min(mGeometryDiagnostics.minFogFactor, fogFactor);
                mGeometryDiagnostics.maxFogFactor =
                    std::max(mGeometryDiagnostics.maxFogFactor, fogFactor);
            }
        }
        if (bypassFog) {
            mGeometryDiagnostics.fogBypassTriangles++;
        }
        if (comb->usedTextures[0] || comb->usedTextures[1]) {
            mGeometryDiagnostics.texturedTriangles++;
        }
        if (comb->usedTextures[0]) {
            if (mRenderingState.mTextures[0] != nullptr) {
                mGeometryDiagnostics.texture0BoundTriangles++;
            } else {
                mGeometryDiagnostics.texture0MissingTriangles++;
            }
        }
        if (forceSimpleMaterial) {
            mGeometryDiagnostics.forcedSimpleMaterialTriangles++;
        }
    }

    struct ShaderProgram* prg = comb->prg[tm];
    if (prg == NULL) {
        comb->prg[tm] = prg =
            LookupOrCreateShaderProgram(comb->shader_id0, comb->shader_id1 | tm * SHADER_OPT(TEXEL0_CLAMP_S));
    }
    if (prg != mRenderingState.mShaderProgram) {
        Flush();
        mRapi->UnloadShader(mRenderingState.mShaderProgram);
        mRapi->LoadShader(prg);
        mRenderingState.mShaderProgram = prg;
    }
    if (use_alpha != mRenderingState.alpha_blend) {
        Flush();
        mRapi->SetUseAlpha(use_alpha);
        mRenderingState.alpha_blend = use_alpha;
    }
    uint8_t numInputs;
    bool usedTextures[2];

    mRapi->ShaderGetInfo(prg, &numInputs, usedTextures);

    struct GfxClipParameters clip_parameters = mRapi->GetClipParameters();

    // [pit-uv-probe]: the PIT strip imports and draws with correct state yet renders
    // flat tan. Log per-vertex (s,t) plus the tile window for tiles 1-4 so a flat
    // draw shows whether the UVs collapsed or landed outside the window.
    static int sPitUvProbeTris = 0;
    const bool pitUvProbeThisTri = gGdxRaceActive != 0 && !is_rect && sPitUvProbeTris < 24 &&
                                    effective_tile[0] >= 1 && effective_tile[0] <= 4;
    if (pitUvProbeThisTri) {
        ++sPitUvProbeTris;
    }

    for (int i = 0; i < 3; i++) {
        float z = v_arr[i]->z, w = v_arr[i]->w;
        if (clip_parameters.z_is_from_0_to_1) {
            z = (z + w) / 2.0f;
        }

        mBufVbo[mBufVboLen++] = v_arr[i]->x;
        mBufVbo[mBufVboLen++] = clip_parameters.invertY ? -v_arr[i]->y : v_arr[i]->y;
        mBufVbo[mBufVboLen++] = z;
        mBufVbo[mBufVboLen++] = w;

        for (int t = 0; t < 2; t++) {
            if (!usedTextures[t]) {
                continue;
            }
            float u = v_arr[i]->u / 32.0f;
            float v = v_arr[i]->v / 32.0f;

            uint32_t uv_tile = effective_tile[t];
            int shifts = mRdp->texture_tile[uv_tile].shifts;
            int shiftt = mRdp->texture_tile[uv_tile].shiftt;

            // Env-gated (GDX_DIAG_PITUV) so a normal Release run stays silent; cached once.
            static const bool sDiagPitUv = std::getenv("GDX_DIAG_PITUV") != nullptr;
            if (sDiagPitUv && pitUvProbeThisTri && t == 0) {
                SPDLOG_ERROR("[pit-uv-probe] tri={} vtx={} tile={} s={} t={} uls={} ult={} lrs={} lrt={} "
                             "shifts={} shiftt={}",
                             sPitUvProbeTris, i, uv_tile, (int)v_arr[i]->u, (int)v_arr[i]->v,
                             mRdp->texture_tile[uv_tile].uls, mRdp->texture_tile[uv_tile].ult,
                             mRdp->texture_tile[uv_tile].lrs, mRdp->texture_tile[uv_tile].lrt, shifts, shiftt);
            }

            if (shifts != 0) {
                if (shifts <= 10) {
                    u /= 1 << shifts;
                } else {
                    u *= 1 << (16 - shifts);
                }
            }
            if (shiftt != 0) {
                if (shiftt <= 10) {
                    v /= 1 << shiftt;
                } else {
                    v *= 1 << (16 - shiftt);
                }
            }

            u -= mRdp->texture_tile[uv_tile].uls / 4.0f;
            v -= mRdp->texture_tile[uv_tile].ult / 4.0f;

            // Must agree with the sampler filter policy above (only G_TF_BILERP is linear).
            if ((mRdp->other_mode_h & (3U << G_MDSFT_TEXTFILT)) == G_TF_BILERP) {
                // Linear filtering shifts an N64 texel index to a GPU texel centre. For rects the
                // offset is opt-in; see sGdxRectHalfTexelDisabled. File-scope flag rather than a
                // function-local static so this per-vertex path pays no init guard.
                if (!is_rect || !sGdxRectHalfTexelDisabled) {
                    u += 0.5f;
                    v += 0.5f;
                }
            }

            const float normalizedU = u / tex_width[t];
            const float normalizedV = v / tex_height[t];
            // An axis with CLAMP enabled should never receive coordinates several repeats outside
            // its tile window. Wrap-masked tiles legitimately do and no longer trigger.
            static const bool diagUvProbe = std::getenv("GDX_DIAG_UVPROBE") != nullptr;
            const uint8_t probeCms = mRdp->texture_tile[uv_tile].cms;
            const uint8_t probeCmt = mRdp->texture_tile[uv_tile].cmt;
            const bool clampAnomaly =
                (((probeCmt & G_TX_CLAMP) != 0) && (normalizedV < -2.0f || normalizedV > 2.0f)) ||
                (((probeCms & G_TX_CLAMP) != 0) && (normalizedU < -2.0f || normalizedU > 2.0f));
            if (diagUvProbe && gGdxRaceActive != 0 && !is_rect && t == 0 && clampAnomaly) {
                static int sUvProbe = 0;
                if (sUvProbe < 96) {
                    ++sUvProbe;
                    FILE* uf = fopen("uvprobe.txt", "a");
                    if (uf != nullptr) {
                        fprintf(uf,
                                "uv_tile=%u rawS=%d rawT=%d shifts=%d shiftt=%d "
                                "texW=%u texH=%u tileW2=%u tileH2=%u u=%.2f v=%.2f normU=%.2f normV=%.2f "
                                "cms=%u cmt=%u masks=%u maskt=%u size=[%u,%u..%u,%u] "
                                "scale=%04X/%04X line=%u tmem=0x%X fmt=%u siz=%u "
                                "addr=%p combine=%016llX geom=%08X othermodeH=%08X\n",
                                uv_tile, (int)v_arr[i]->u, (int)v_arr[i]->v, shifts, shiftt,
                                tex_width[t], tex_height[t], tex_width2[t], tex_height2[t],
                                u, v, normalizedU, normalizedV,
                                mRdp->texture_tile[uv_tile].cms, mRdp->texture_tile[uv_tile].cmt,
                                mRdp->texture_tile[uv_tile].masks, mRdp->texture_tile[uv_tile].maskt,
                                mRdp->texture_tile[uv_tile].uls, mRdp->texture_tile[uv_tile].ult,
                                mRdp->texture_tile[uv_tile].lrs, mRdp->texture_tile[uv_tile].lrt,
                                (unsigned)mRsp->texture_scaling_factor.s,
                                (unsigned)mRsp->texture_scaling_factor.t,
                                mRdp->texture_tile[uv_tile].line_size_bytes,
                                (unsigned)mRdp->texture_tile[uv_tile].tmem,
                                (unsigned)mRdp->texture_tile[uv_tile].fmt,
                                (unsigned)mRdp->texture_tile[uv_tile].siz,
                                (const void*)mRdp->loaded_texture[mRdp->texture_tile[uv_tile].tmem_index].addr,
                                (unsigned long long)mRdp->combine_mode,
                                (unsigned)mRsp->geometry_mode,
                                (unsigned)mRdp->other_mode_h);
                        fclose(uf);
                    }
                    // Dump the SETTILE ring plus a draw marker so the tile-write
                    // history immediately preceding this suspect draw is visible.
                    if (std::getenv("GDX_DIAG_SETTILE") != nullptr) {
                        extern int gGdxTraceSeq;
                        extern std::deque<std::string> gGdxTileTraceRing;
                        static int sTraceDumps = 0;
                        if (sTraceDumps < 16) {
                            ++sTraceDumps;
                            FILE* tf = fopen("settile-trace.txt", "a");
                            if (tf != nullptr) {
                                fprintf(tf, "==== dump %d ====\n", sTraceDumps);
                                for (const std::string& lineStr : gGdxTileTraceRing) {
                                    fprintf(tf, "%s\n", lineStr.c_str());
                                }
                                fprintf(tf,
                                        "D %06d tile=%u normV=%.2f cms=%u cmt=%u masks=%u maskt=%u addr=%p "
                                        "size=[%u,%u..%u,%u] pos=(%.0f,%.0f,%.0f,%.0f)\n",
                                        ++gGdxTraceSeq, uv_tile, normalizedV,
                                        mRdp->texture_tile[uv_tile].cms, mRdp->texture_tile[uv_tile].cmt,
                                        mRdp->texture_tile[uv_tile].masks, mRdp->texture_tile[uv_tile].maskt,
                                        (const void*)mRdp->loaded_texture[mRdp->texture_tile[uv_tile].tmem_index].addr,
                                        mRdp->texture_tile[uv_tile].uls, mRdp->texture_tile[uv_tile].ult,
                                        mRdp->texture_tile[uv_tile].lrs, mRdp->texture_tile[uv_tile].lrt,
                                        v_arr[i]->x, v_arr[i]->y, v_arr[i]->z, v_arr[i]->w);
                                fclose(tf);
                            }
                            gGdxTileTraceRing.clear();
                        }
                    }
                }
            }
            mBufVbo[mBufVboLen++] = normalizedU;
            mBufVbo[mBufVboLen++] = normalizedV;
            if (!is_rect && t == 0 && std::isfinite(normalizedU) && std::isfinite(normalizedV)) {
                mGeometryDiagnostics.minTextureU = std::min(mGeometryDiagnostics.minTextureU, normalizedU);
                mGeometryDiagnostics.maxTextureU = std::max(mGeometryDiagnostics.maxTextureU, normalizedU);
                mGeometryDiagnostics.minTextureV = std::min(mGeometryDiagnostics.minTextureV, normalizedV);
                mGeometryDiagnostics.maxTextureV = std::max(mGeometryDiagnostics.maxTextureV, normalizedV);
            }

            bool clampS = tm & (1 << 2 * t);
            bool clampT = tm & (1 << 2 * t + 1);

            if (clampS) {
                mBufVbo[mBufVboLen++] = (tex_width2[t] - 0.5f) / tex_width[t];
            }

            if (clampT) {
                mBufVbo[mBufVboLen++] = (tex_height2[t] - 0.5f) / tex_height[t];
            }
        }

        if (use_fog) {
            if (use_blend_color) {
                mBufVbo[mBufVboLen++] = mRdp->blend_color.r / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->blend_color.g / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->blend_color.b / 255.0f;
                mBufVbo[mBufVboLen++] = blendFactor;
            } else {
                mBufVbo[mBufVboLen++] = mRdp->fog_color.r / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->fog_color.g / 255.0f;
                mBufVbo[mBufVboLen++] = mRdp->fog_color.b / 255.0f;
                mBufVbo[mBufVboLen++] = v_arr[i]->color.a / 255.0f; // fog factor (not alpha)
            }
        }

        if (use_grayscale) {
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.r / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.g / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.b / 255.0f;
            mBufVbo[mBufVboLen++] = mRdp->grayscale_color.a / 255.0f; // lerp interpolation factor (not alpha)
        }

        for (int j = 0; j < numInputs; j++) {
            RGBA* color;
            RGBA tmp;
            for (int k = 0; k < 1 + (use_alpha ? 1 : 0); k++) {
                switch (comb->shader_input_mapping[k][j]) {
                        // Note: CCMUX constants and ACMUX constants used here have same value, which is why this works
                        // (except LOD fraction).
                    case G_CCMUX_PRIMITIVE:
                        color = &mRdp->prim_color;
                        break;
                    case G_CCMUX_SHADE:
                        color = &v_arr[i]->color;
                        break;
                    case G_CCMUX_ENVIRONMENT:
                        color = &mRdp->env_color;
                        break;
                    case G_CCMUX_PRIMITIVE_ALPHA: {
                        tmp.r = tmp.g = tmp.b = mRdp->prim_color.a;
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_ENV_ALPHA: {
                        tmp.r = tmp.g = tmp.b = mRdp->env_color.a;
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_PRIM_LOD_FRAC: {
                        tmp.r = tmp.g = tmp.b = mRdp->prim_lod_fraction;
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_LOD_FRACTION: {
                        if (mRdp->other_mode_l & G_TL_LOD) {
                            // "Hack" that works for Bowser - Peach painting
                            float distance_frac = (v1->w - 3000.0f) / 3000.0f;
                            if (distance_frac < 0.0f) {
                                distance_frac = 0.0f;
                            }
                            if (distance_frac > 1.0f) {
                                distance_frac = 1.0f;
                            }
                            tmp.r = tmp.g = tmp.b = tmp.a = distance_frac * 255.0f;
                        } else {
                            tmp.r = tmp.g = tmp.b = tmp.a = 255.0f;
                        }
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_KEY_CENTER:
                        color = &mRdp->key_center;
                        break;
                    case G_CCMUX_KEY_SCALE:
                        color = &mRdp->key_scale;
                        break;
                    case G_CCMUX_CONVERT_K4: {
                        tmp.r = tmp.g = tmp.b = mRdp->convert_k[4];
                        color = &tmp;
                        break;
                    }
                    case G_CCMUX_CONVERT_K5: {
                        tmp.r = tmp.g = tmp.b = mRdp->convert_k[5];
                        color = &tmp;
                        break;
                    }
                    case G_ACMUX_PRIM_LOD_FRAC:
                        tmp.a = mRdp->prim_lod_fraction;
                        color = &tmp;
                        break;
                    default:
                        memset(&tmp, 0, sizeof(tmp));
                        color = &tmp;
                        break;
                }
                if (k == 0) {
                    mBufVbo[mBufVboLen++] = color->r / 255.0f;
                    mBufVbo[mBufVboLen++] = color->g / 255.0f;
                    mBufVbo[mBufVboLen++] = color->b / 255.0f;
                } else {
                    if (use_fog && !use_blend_color && color == &v_arr[i]->color) {
                        // Shade alpha is 100% for standard fog, blend color mode preserves
                        // it since fog alpha is the blend factor
                        mBufVbo[mBufVboLen++] = 1.0f;
                    } else {
                        mBufVbo[mBufVboLen++] = color->a / 255.0f;
                    }
                }
            }
        }

        // struct RGBA *color = &v_arr[i]->color;
        // mBufVbo[mBufVboLen++] = color->r / 255.0f;
        // mBufVbo[mBufVboLen++] = color->g / 255.0f;
        // mBufVbo[mBufVboLen++] = color->b / 255.0f;
        // mBufVbo[mBufVboLen++] = color->a / 255.0f;
    }

    if (!is_rect) {
        mGeometryDiagnostics.trianglesEmitted++;
    }
    if (++mBufVboNumTris == MAX_TRI_BUFFER) {
        // if (++mBufVbo_num_tris == 1) {
        Flush();
    }
}

void Interpreter::GfxSpGeometryMode(uint32_t clear, uint32_t set) {
    mRsp->geometry_mode &= ~clear;
    mRsp->geometry_mode |= set;
}

void Interpreter::GfxSpExtraGeometryMode(uint32_t clear, uint32_t set) {
    mRsp->extra_geometry_mode &= ~clear;
    mRsp->extra_geometry_mode |= set;
}

// Expand an L3DEX2 G_LINE3D into a screen-space quad: Fast3D has no line primitive, so the port's
// gfx bridge rewrites Course Edit's line commands into OTR_G_LINE3D_GDX and this draws each as two
// triangles of constant pixel width from the two already-transformed vertices.
void Interpreter::GfxSpLine3DGdx(uint8_t vtx1Idx, uint8_t vtx2Idx, uint8_t halfWidth) {
    const struct LoadedVertex a = mRsp->loaded_vertices[vtx1Idx];
    const struct LoadedVertex b = mRsp->loaded_vertices[vtx2Idx];

    /* Whether the handler fires in Course Edit at all, and whether the w <= 0 early-out is what
       suppresses the spline lines. */
    {
        static int sGdxLineDbg = 0;
        if (sGdxLineDbg < 32) {
            sGdxLineDbg++;
            gdx_dbg_logf("[GDX-DBG line] v0=%u v1=%u wd=%u aw=%.3f bw=%.3f %s\n",
                         vtx1Idx, vtx2Idx, halfWidth, a.w, b.w,
                         (a.w <= 0.0f || b.w <= 0.0f) ? "EARLYOUT(w<=0)" : "draw");
        }
    }

    // Both endpoints must be in front of the camera; near-plane clipping of partially
    // visible lines is not worth the complexity for the editor's on-canvas geometry.
    if (a.w <= 0.0f || b.w <= 0.0f) {
        return;
    }

    const float ax = a.x / a.w, ay = a.y / a.w;
    const float bx = b.x / b.w, by = b.y / b.w;

    // Direction in pixel space so the visual width is uniform regardless of line angle
    // and window aspect.
    const float pixelHalfW = (float)mCurDimensions.width * 0.5f;
    const float pixelHalfH = (float)mCurDimensions.height * 0.5f;
    const float dx = (bx - ax) * pixelHalfW;
    const float dy = (by - ay) * pixelHalfH;
    const float len = sqrtf(dx * dx + dy * dy);
    if (len < 0.0001f) {
        return;
    }
    // Hardware line width: wd is in half-pixel units on top of a 1.5px base, and the
    // quad extends half that width to each side. Scale from the 320x240 native raster
    // to the current output so lines keep their authored proportion.
    const float nativeWidthPx = 1.5f + (float)halfWidth * 0.5f;
    const float outputScale = (float)mCurDimensions.height / 240.0f;
    const float halfPx = nativeWidthPx * 0.5f * (outputScale > 0.0f ? outputScale : 1.0f);
    const float perpX = (-dy / len) * halfPx / pixelHalfW; // back to NDC units
    const float perpY = (dx / len) * halfPx / pixelHalfH;

    struct LoadedVertex* quad = &mRsp->loaded_vertices[MAX_VERTICES + 0];
    quad[0] = a;
    quad[1] = a;
    quad[2] = b;
    quad[3] = b;
    quad[0].x = (ax + perpX) * a.w;
    quad[0].y = (ay + perpY) * a.w;
    quad[1].x = (ax - perpX) * a.w;
    quad[1].y = (ay - perpY) * a.w;
    quad[2].x = (bx + perpX) * b.w;
    quad[2].y = (by + perpY) * b.w;
    quad[3].x = (bx - perpX) * b.w;
    quad[3].y = (by - perpY) * b.w;
    quad[0].clip_rej = 0;
    quad[1].clip_rej = 0;
    quad[2].clip_rej = 0;
    quad[3].clip_rej = 0;

    // Hardware lines never face-cull; suppress culling for the quad only.
    const uint32_t savedGeometryMode = mRsp->geometry_mode;
    mRsp->geometry_mode &= ~get_attr(CULL_BOTH);
    GfxSpTri1(MAX_VERTICES + 0, MAX_VERTICES + 1, MAX_VERTICES + 2, false);
    GfxSpTri1(MAX_VERTICES + 1, MAX_VERTICES + 3, MAX_VERTICES + 2, false);
    mRsp->geometry_mode = savedGeometryMode;
}

void Interpreter::AdjustVIewportOrScissor(XYWidthHeight* area) {
    if (!mFbActive) {
        // Adjust the y origin based on the y-inversion for the active framebuffer
        GfxClipParameters clipParameters = mRapi->GetClipParameters();
        if (clipParameters.invertY) {
            area->y -= area->height;
        } else {
            area->y = mNativeDimensions.height - area->y;
        }

        area->width *= RATIO_X(mActiveFrameBuffer, mCurDimensions);
        area->height *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);
        area->x *= RATIO_X(mActiveFrameBuffer, mCurDimensions);
        area->y *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);

        if (!mRendersToFb || (mMsaaLevel > 1 && mCurDimensions.width == mGameWindowViewport.width &&
                              mCurDimensions.height == mGameWindowViewport.height)) {
            area->x += mGameWindowViewport.x;
            area->y += mGfxCurrentWindowDimensions.height - (mGameWindowViewport.y + mGameWindowViewport.height);
        }
    } else {
        area->y = mActiveFrameBuffer->second.orig_height - area->y;

        if (mActiveFrameBuffer->second.resize) {
            area->width *= RATIO_X(mActiveFrameBuffer, mCurDimensions);
            area->height *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);
            area->x *= RATIO_X(mActiveFrameBuffer, mCurDimensions);
            area->y *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);
        }
    }
}

void Interpreter::CalcAndSetViewport(const F3DVp_t* viewport) {
    // 2 bits fraction
    float width = 2.0f * viewport->vscale[0] / 4.0f;
    float height = 2.0f * viewport->vscale[1] / 4.0f;
    float x = (viewport->vtrans[0] / 4.0f) - width / 2.0f;
    float y = ((viewport->vtrans[1] / 4.0f) + height / 2.0f);

    mRdp->viewport.x = x;
    mRdp->viewport.y = y;
    mRdp->viewport.width = width;
    mRdp->viewport.height = height;
    mRsp->viewport_z_scale = viewport->vscale[2];
    mRsp->viewport_z_trans = viewport->vtrans[2];

    AdjustVIewportOrScissor(&mRdp->viewport);

    mRdp->viewport_or_scissor_changed = true;
}

void Interpreter::GfxSpMovememF3dex2(uint8_t index, uint8_t offset, const void* data) {
    switch (index) {
        case F3DEX2_G_MV_VIEWPORT:
            CalcAndSetViewport((const F3DVp_t*)data);
            break;
        case F3DEX2_G_MV_LIGHT: {
            // F3DFLX2.Rej repurposes the LOOKATY slot as its reflection-alpha light, but only
            // after the game DMAs the alpha LUT into DMEM via G_DMA_IO. Menu tasks run the FLX
            // variant too and upload a genuine LookAt Y there; without the LUT gate that upload
            // was consumed and lookat[1] never updated, breaking texture-gen reflections.
            if (mF3dex2Variant == F3dex2Variant::FZeroFlxReject && offset == 24 && data != nullptr &&
                mRsp->dma_io_loaded) {
                // The alpha light is host-written GfxPool data (racer.c),
                // so its s16 direction fields are native-endian, not the
                // big-endian layout ROM data would have.
                const uint8_t* bytes = static_cast<const uint8_t*>(data);
                int16_t dir[3];
                memcpy(dir, bytes + 8, sizeof(dir));
                mRsp->f3dflx_alpha_light[0] = dir[0] / 256.0f;
                mRsp->f3dflx_alpha_light[1] = dir[1] / 256.0f;
                mRsp->f3dflx_alpha_light[2] = dir[2] / 256.0f;
                NormalizeVector(mRsp->f3dflx_alpha_light);
                mRsp->f3dflx_alpha_light_valid = true;
                break;
            }
            int lightidx = offset / 24 - 2;
            if (lightidx >= 0 && lightidx <= MAX_LIGHTS) { // skip lookat
                // NOTE: reads out of bounds if it is an ambient light
                memcpy(mRsp->current_lights + lightidx, data, sizeof(F3DLight));
            } else if (lightidx < 0) {
                memcpy(mRsp->lookat + offset / 24, data, sizeof(F3DLight_t)); // TODO Light?
            }
            break;
        }
    }
}

void Interpreter::GfxSpDmaIo(bool write, uint16_t dmem, void* data, size_t size) {
    if (data == nullptr || dmem >= sizeof(mRsp->dmem)) {
        return;
    }
    const size_t copySize = std::min(size, sizeof(mRsp->dmem) - static_cast<size_t>(dmem));
    if (write) {
        memcpy(data, mRsp->dmem + dmem, copySize);
    } else {
        memcpy(mRsp->dmem + dmem, data, copySize);
        mRsp->dma_io_dmem = dmem;
        mRsp->dma_io_loaded = true;
        mGeometryDiagnostics.dmaIoLoads++;
    }
}

void Interpreter::GfxSpMovememF3d(uint8_t index, uint8_t offset, const void* data) {
    switch (index) {
        case F3DEX_G_MV_VIEWPORT:
            CalcAndSetViewport((const F3DVp_t*)data);
            break;
        case F3DEX_G_MV_LOOKATY:
        case F3DEX_G_MV_LOOKATX:
            memcpy(mRsp->lookat + (index - F3DEX_G_MV_LOOKATY) / 2, data, sizeof(F3DLight_t));
            break;
        case F3DEX_G_MV_L0:
        case F3DEX_G_MV_L1:
        case F3DEX_G_MV_L2:
        case F3DEX_G_MV_L3:
        case F3DEX_G_MV_L4:
        case F3DEX_G_MV_L5:
        case F3DEX_G_MV_L6:
        case F3DEX_G_MV_L7:
            // NOTE: reads out of bounds if it is an ambient light
            memcpy(mRsp->current_lights + (index - F3DEX_G_MV_L0) / 2, data, sizeof(F3DLight_t));
            break;
    }
}

void Interpreter::GfxSpMovewordF3dex2(uint8_t index, uint16_t offset, uintptr_t data) {
    switch (index) {
        case G_MW_NUMLIGHT:
            mRsp->current_num_lights = data / 24 + 1; // add ambient light
            mRsp->lights_changed = true;
            break;
        case G_MW_FOG:
            mRsp->fog_mul = (int16_t)(data >> 16);
            mRsp->fog_offset = (int16_t)data;
            break;
        case G_MW_SEGMENT: {
            int segNumber = offset / 4;
            mSegmentPointers[segNumber] = data;
        } break;
        case G_MW_SEGMENT_INTERP: {
            int segNumber = offset % 16;
            int segIndex = offset / 16;

            if (segIndex == mInterpolationIndex)
                mSegmentPointers[segNumber] = data;
        } break;
    }
}

void Interpreter::GfxSpMovewordF3d(uint8_t index, uint16_t offset, uintptr_t data) {
    switch (index) {
        case G_MW_NUMLIGHT:
            // Ambient light is included
            // The 31st bit is a flag that lights should be recalculated
            mRsp->current_num_lights = (data - 0x80000000U) / 32;
            mRsp->lights_changed = true;
            break;
        case G_MW_FOG:
            mRsp->fog_mul = (int16_t)(data >> 16);
            mRsp->fog_offset = (int16_t)data;
            break;
        case G_MW_SEGMENT: {
            int segNumber = offset / 4;
            mSegmentPointers[segNumber] = data;
        } break;
        case G_MW_SEGMENT_INTERP: {
            int segNumber = offset % 16;
            int segIndex = offset / 16;

            if (segIndex == mInterpolationIndex)
                mSegmentPointers[segNumber] = data;
        } break;
    }
}

void Interpreter::GfxSpTexture(uint16_t sc, uint16_t tc, uint8_t level, uint8_t tile, uint8_t on) {
    mRsp->texture_scaling_factor.s = sc;
    mRsp->texture_scaling_factor.t = tc;
    if (mRdp->first_tile_index != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }

    mRdp->first_tile_index = tile;
}

void Interpreter::GfxDpSetScissor(uint32_t mode, uint32_t ulx, uint32_t uly, uint32_t lrx, uint32_t lry) {
    // F-Zero X frames every screen inside the CRT-overscan-safe area via two scissors hardcoded at
    // ~50 call sites ({12,16,308,224} and {12,8,308,232}); intercepting them here is the one
    // chokepoint that also covers static DLs. Exact match only, so split-screen boxes and menu
    // reveal wipes never widen.
    // VR eye targets always drop them: the eye's own field of view is the frame edge, and a fixed
    // border would sit in the viewer's periphery as a black frame.
    if ((mRemoveBordersCache || mVrEyeActive) && ulx == (12 << 2) && lrx == (308 << 2) &&
        ((uly == (16 << 2) && lry == (224 << 2)) || (uly == (8 << 2) && lry == (232 << 2)))) {
        ulx = 0;
        uly = 0;
        lrx = 320 << 2;
        lry = 240 << 2;
    }
    float x = ulx / 4.0f;
    float y = lry / 4.0f;
    float width = (lrx - ulx) / 4.0f;
    float height = (lry - uly) / 4.0f;

    mRdp->scissor.x = x;
    mRdp->scissor.y = y;
    mRdp->scissor.width = width;
    mRdp->scissor.height = height;

    AdjustVIewportOrScissor(&mRdp->scissor);

    mRdp->viewport_or_scissor_changed = true;
}

void Interpreter::GfxDpSetTextureImage(uint32_t format, uint32_t size, uint32_t width, const char* texPath,
                                       uint32_t texFlags, RawTexMetadata rawTexMetdata, const void* addr) {
    // fprintf(stderr, "GfxDpSetTextureImage: %s (width=%d; size=0x%X)\n",
    //         rawTexMetdata.resource ? rawTexMetdata.resource->GetInitData()->Path.c_str() : nullptr, width, size);
    mRdp->texture_to_load.addr = (const uint8_t*)addr;
    mRdp->texture_to_load.siz = size;
    mRdp->texture_to_load.width = width;
    mRdp->texture_to_load.tex_flags = texFlags;
    mRdp->texture_to_load.raw_tex_metadata = rawTexMetdata;
}

void Interpreter::GfxDpSetTile(uint8_t fmt, uint32_t siz, uint32_t line, uint32_t tmem, uint8_t tile, uint32_t palette,
                               uint32_t cmt, uint32_t maskt, uint32_t shiftt, uint32_t cms, uint32_t masks,
                               uint32_t shifts) {
    // OTRTODO:
    // SUPPORT_CHECK(tmem == 0 || tmem == 256);

    // Trace tile-descriptor writes so stale-state bugs (a draw inheriting another
    // object's SETTILE) can be correlated against draw markers in the same file.
    static const bool diagSetTile = std::getenv("GDX_DIAG_SETTILE") != nullptr;
    if (diagSetTile) {
        extern int gGdxTraceSeq;
        extern void GdxTileTracePush(const char* line);
        char buf[160];
        snprintf(buf, sizeof(buf), "S %06d tile=%u fmt=%u siz=%u line=%u tmem=0x%X cms=%u masks=%u cmt=%u maskt=%u",
                 ++gGdxTraceSeq, tile, fmt, siz, line, tmem, cms, masks, cmt, maskt);
        GdxTileTracePush(buf);
        // The course setup DL is the only source of masks==7 tile writes; log those
        // executions directly (bypassing the ring) to prove whether they ever run.
        if (masks == 7 && gGdxRaceActive != 0) {
            static int sWideTileCount = 0;
            if (sWideTileCount < 64) {
                ++sWideTileCount;
                FILE* wf = fopen("settile-trace.txt", "a");
                if (wf != nullptr) {
                    fprintf(wf, "W %06d %s\n", gGdxTraceSeq, buf);
                    fclose(wf);
                }
            }
        }
    }

    if (cms == G_TX_WRAP && masks == G_TX_NOMASK) {
        cms = G_TX_CLAMP;
    }
    if (cmt == G_TX_WRAP && maskt == G_TX_NOMASK) {
        cmt = G_TX_CLAMP;
    }

    mRdp->texture_tile[tile].palette = palette; // palette should set upper 4 bits of color index in 4b mode
    mRdp->texture_tile[tile].fmt = fmt;
    mRdp->texture_tile[tile].siz = siz;
    mRdp->texture_tile[tile].cms = cms;
    mRdp->texture_tile[tile].cmt = cmt;
    mRdp->texture_tile[tile].masks = masks;
    mRdp->texture_tile[tile].maskt = maskt;
    mRdp->texture_tile[tile].shifts = shifts;
    mRdp->texture_tile[tile].shiftt = shiftt;
    mRdp->texture_tile[tile].line_size_bytes = line * 8;

    mRdp->texture_tile[tile].tmem = tmem;
    mRdp->texture_tile[tile].tmem_index = static_cast<uint16_t>(tmem);

    mRdp->textures_changed[0] = true;
    mRdp->textures_changed[1] = true;
}

void Interpreter::GfxDpSetTileSize(uint8_t tile, uint16_t uls, uint16_t ult, uint16_t lrs, uint16_t lrt) {
    mRdp->texture_tile[tile].uls = uls;
    mRdp->texture_tile[tile].ult = ult;
    mRdp->texture_tile[tile].lrs = lrs;
    mRdp->texture_tile[tile].lrt = lrt;
    mRdp->textures_changed[0] = true;
    mRdp->textures_changed[1] = true;
}

void Interpreter::GfxDpLoadTlut(uint8_t tile, uint32_t high_index) {
    SUPPORT_CHECK(mRdp->texture_to_load.siz == G_IM_SIZ_16b);

    uint16_t tmem = mRdp->texture_tile[tile].tmem;
    const uint8_t* src = mRdp->texture_to_load.addr;
    uint32_t entryCount = high_index + 1;
    uint32_t byteCount = entryCount * 2;

    // Record every address ever bound as a TLUT source: the O(1) gate that keeps
    // TextureCacheDeletePalette from scanning the whole cache on an ordinary texture refresh.
    if (src != nullptr) {
        mSeenPaletteAddrs.insert(src);
    }

    if (tmem >= 256) {
        // N64 TMEM palette area starts at tmem word 256. Each CI4 palette = 16 entries = 16 tmem words.
        uint32_t paletteByteOffset = (tmem - 256) * 2;

        if (high_index == 255 && paletteByteOffset == 0) {
            // CI8: full 256-entry palette spanning both halves
            memcpy(mRdp->palette_staging[0], src, 256);
            memcpy(mRdp->palette_staging[1], src + 256, 256);
            mRdp->palettes[0] = mRdp->palette_staging[0];
            mRdp->palettes[1] = mRdp->palette_staging[1];
            mRdp->palette_dram_addr[0] = src;
            mRdp->palette_dram_addr[1] = src + 256;
        } else if (paletteByteOffset < 256) {
            // Palettes 0-7 range
            uint32_t copyLen = (paletteByteOffset + byteCount <= 256) ? byteCount : (256 - paletteByteOffset);
            memcpy(mRdp->palette_staging[0] + paletteByteOffset, src, copyLen);
            mRdp->palettes[0] = mRdp->palette_staging[0];
            mRdp->palette_dram_addr[0] = src;
        } else {
            // Palettes 8-15 range
            uint32_t offset = paletteByteOffset - 256;
            uint32_t copyLen = (offset + byteCount <= 256) ? byteCount : (256 - offset);
            memcpy(mRdp->palette_staging[1] + offset, src, copyLen);
            mRdp->palettes[1] = mRdp->palette_staging[1];
            mRdp->palette_dram_addr[1] = src;
        }
    } else {
        // tmem < 256: non-standard location, fall back to direct pointer
        mRdp->palettes[1] = src;
        mRdp->palette_dram_addr[1] = src;
    }
}

void Interpreter::StoreLoadedTexture(uint16_t tmemStart, const LoadedTexture& texture) {
    constexpr uint32_t kTmemWordBytes = 8;
    constexpr uint32_t kTmemWordCount = 512;
    if (tmemStart >= kTmemWordCount) {
        return;
    }

    // Live getenv, not static-once: the bridge exports GDX_DIAG_SETTIMG via _putenv at the first
    // real GFX task, and boot frames import textures before that.
    const bool sTmemTrace = std::getenv("GDX_DIAG_SETTIMG") != nullptr;
    if (sTmemTrace && gGdxRaceActive != 0) { // race-gated: boot stores ate the whole budget otherwise
        static int sTmemStoreLogs = 0;
        if (sTmemStoreLogs < 400) {
            ++sTmemStoreLogs;
            SPDLOG_ERROR("[tmem] store start={} sizeB={} origB={} addr={}", tmemStart,
                         texture.size_bytes, texture.orig_size_bytes, fmt::ptr(texture.addr));
        }
    }

    const uint32_t logicalSize = texture.orig_size_bytes != 0 ? texture.orig_size_bytes : texture.size_bytes;
    const uint64_t span = texture.resource_tmem_line_bytes != 0
        ? std::max<uint64_t>(logicalSize, uint64_t(texture.resource_tmem_line_bytes) * texture.resource_height)
        : logicalSize;
    const uint32_t requestedWords = static_cast<uint32_t>(std::min<uint64_t>(512, std::max<uint64_t>(1, (span + 7) / 8)));
    const uint32_t wordCount = std::min(requestedWords, kTmemWordCount - tmemStart);
    const uint32_t newEnd = tmemStart + wordCount;

    // A TMEM upload replaces bytes, not a named texture object. Discard any
    // older metadata range touched by this upload so a later render tile can
    // never select stale DRAM data for overwritten TMEM.
    for (auto& existing : mRdp->loaded_texture) {
        if (existing.tmem_word_count == 0) {
            continue;
        }
        const uint32_t existingEnd = existing.tmem_start + existing.tmem_word_count;
        if (existing.tmem_start < newEnd && tmemStart < existingEnd) {
            existing = {};
        }
    }

    // Materialize metadata at every covered TMEM word. Render tiles are legal
    // at offsets inside a LOADBLOCK/LOADTILE range, not only at its base.
    for (uint32_t word = 0; word < wordCount; ++word) {
        const uint32_t logicalByteOffset = word * kTmemWordBytes;
        LoadedTexture entry = texture;
        entry.tmem_start = tmemStart;
        entry.tmem_word_count = static_cast<uint16_t>(wordCount);
        if (texture.resource_tmem_line_bytes != 0) {
            const uint32_t row = logicalByteOffset / texture.resource_tmem_line_bytes;
            const uint32_t column = (logicalByteOffset % texture.resource_tmem_line_bytes) * 8 /
                                    ScaledTextureBitsPerPixel(texture.raw_tex_metadata);
            if (row >= texture.resource_height || column >= texture.resource_width ||
                !SetScaledTextureView(entry, texture.addr, column, row, texture.resource_width - column,
                                    texture.resource_height - row, texture.resource_tmem_line_bytes)) {
                entry.addr = nullptr;
                entry.size_bytes = 0;
            }
            entry.orig_size_bytes = texture.orig_size_bytes > logicalByteOffset
                ? texture.orig_size_bytes - logicalByteOffset : 0;
            mRdp->loaded_texture[tmemStart + word] = entry;
            continue;
        }
        const uint32_t hostByteOffset =
            logicalSize != 0 ? static_cast<uint32_t>((static_cast<uint64_t>(logicalByteOffset) * texture.size_bytes) /
                                                     logicalSize)
                             : logicalByteOffset;
        if (entry.addr != nullptr) {
            entry.addr += hostByteOffset;
        }
        entry.size_bytes = texture.size_bytes > hostByteOffset ? texture.size_bytes - hostByteOffset : 0;
        entry.orig_size_bytes =
            texture.orig_size_bytes > logicalByteOffset ? texture.orig_size_bytes - logicalByteOffset : 0;
        if (word != 0) {
            // An interior TMEM address starts a new render view into this upload. Its row layout
            // comes from the render tile, not the load base; keeping the base stride collapses
            // these views into one-pixel-high stripes.
            entry.line_size_bytes = 0;
            entry.full_image_line_size_bytes = 0;
        }
        mRdp->loaded_texture[tmemStart + word] = entry;
    }
}

// Lookup needs the band's native offset in its parent atlas. A replacement owns tightly packed
// rows, independent of that parent's stride. CI bands have no runtime palette side-channel.
static bool GdxTryAtlasTileOverride(RDP* rdp, LoadedTexture& loaded, uint8_t loadTile, uint32_t bandWidth,
                                    uint32_t bandHeight, bool block) {
    if (loaded.addr == nullptr || loaded.raw_tex_metadata.resource != nullptr || bandWidth == 0 || bandHeight == 0) {
        return false;
    }
    const uint8_t fmt = rdp->texture_tile[loadTile].fmt;
    const uint8_t siz = rdp->texture_to_load.siz;
    if (fmt == G_IM_FMT_CI) {
        return false;
    }
    if (!gdx_workshop_texture_packs_enabled()) {
        return false;
    }
    size_t byteOffset = 0;
    const char* baseKey = GDiffuser_LookupLoadedAssetKeyContaining(loaded.addr, &byteOffset);
    if (baseKey == nullptr) {
        return false;
    }
    const char* packPath =
        GdxWorkshopLookupAtlasTileOverride(baseKey, byteOffset, fmt, siz, (int)bandWidth, (int)bandHeight);
    if (packPath == nullptr) {
        return false;
    }
    std::shared_ptr<Fast::Texture> tex = std::static_pointer_cast<Fast::Texture>(
        Ship::Context::GetInstance()->GetResourceManager()->LoadResourceProcess(packPath));
    if (tex == nullptr || tex->ImageData == nullptr) {
        return false;
    }
    LoadedTexture replacement = loaded;
    replacement.raw_tex_metadata = {tex->Width, tex->Height, tex->HByteScale, tex->VPixelScale, tex, tex->Type};
    replacement.resource_bit_offset = 0;
    if (IsScaledLinearTexture(replacement.raw_tex_metadata)) {
        const uint32_t rowBytes = (bandWidth * ScaledTextureBitsPerPixel(replacement.raw_tex_metadata) + 7) / 8;
        const uint32_t tmemLine = block ? rowBytes : (rowBytes + 7) & ~7u;
        if (!SetScaledTextureView(replacement, tex->ImageData, 0, 0, bandWidth, bandHeight, tmemLine)) {
            return false;
        }
        replacement.tex_flags = tex->Flags;
        loaded = std::move(replacement);
        return true;
    }
    loaded.raw_tex_metadata.width = tex->Width;
    loaded.raw_tex_metadata.height = tex->Height;
    loaded.raw_tex_metadata.h_byte_scale = tex->HByteScale;
    loaded.raw_tex_metadata.v_pixel_scale = tex->VPixelScale;
    loaded.raw_tex_metadata.type = tex->Type;
    loaded.raw_tex_metadata.resource = tex;
    loaded.tex_flags = tex->Flags;
    loaded.addr = reinterpret_cast<const uint8_t*>(tex->ImageData);
    loaded.size_bytes =
        static_cast<uint32_t>(loaded.orig_size_bytes * tex->HByteScale * tex->VPixelScale);
    loaded.line_size_bytes = static_cast<uint32_t>(loaded.line_size_bytes * tex->HByteScale);
    // An atlas replacement owns only this band, not the parent atlas's wider rows.
    loaded.full_image_line_size_bytes = loaded.line_size_bytes;
    return true;
}

void Interpreter::LoadScaledTexture(uint8_t tile, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool block) {
    const auto& source = mRdp->texture_to_load;
    LoadedTexture loaded{};
    loaded.raw_tex_metadata = source.raw_tex_metadata;
    loaded.tex_flags = source.tex_flags;
    const uint32_t pixelBits = ScaledTextureBitsPerPixel(source.raw_tex_metadata);
    const uint64_t rowBytes = (uint64_t(width) * pixelBits + 7) / 8;
    const uint64_t bytes = rowBytes * height;
    loaded.orig_size_bytes = static_cast<uint32_t>(std::min<uint64_t>(bytes, UINT32_MAX));
    const uint32_t line = static_cast<uint32_t>(block ? rowBytes : (rowBytes + 7) & ~uint64_t(7));
    uint32_t sx, sy;
    const auto& tex = source.raw_tex_metadata.resource;
    if (ScaledTextureDimensions(source.raw_tex_metadata, sx, sy) && x < tex->Width / sx && y < tex->Height / sy) {
        // Sprite blits request fixed-height TMEM bands, including beyond the image's last row.
        // Keep the transfer span for invalidation but decode only the resource's available pixels.
        width = std::min(width, tex->Width / sx - x);
        height = std::min(height, tex->Height / sy - y);
    }
    const uint8_t logicalSize = pixelBits == 16 ? G_IM_SIZ_16b : pixelBits == 8 ? G_IM_SIZ_8b : G_IM_SIZ_4b;
    const bool compatibleSize = source.siz == logicalSize || (block && source.siz == G_IM_SIZ_16b);
    if (compatibleSize && bytes <= UINT32_MAX) {
        SetScaledTextureView(loaded, source.addr, x, y, width, height, line);
    }
    const auto masked = mMaskedTextures.find(GetBaseTexturePath(source.raw_tex_metadata.resource->GetInitData()->Path));
    if (masked != mMaskedTextures.end()) {
        loaded.masked = true;
        loaded.blended = masked->second.replacementData != nullptr;
    }
    StoreLoadedTexture(mRdp->texture_tile[tile].tmem_index, loaded);
    mRdp->textures_changed[0] = mRdp->textures_changed[1] = true;
}

void Interpreter::GfxDpLoadBlock(uint8_t tile, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t dxt) {
    if (IsScaledLinearTexture(mRdp->texture_to_load.raw_tex_metadata)) {
        uint32_t sx, sy, width = 0;
        const auto& source = mRdp->texture_to_load;
        if (ScaledTextureDimensions(source.raw_tex_metadata, sx, sy)) {
            width = source.width > 1 ? source.width : source.raw_tex_metadata.resource->Width / sx;
        }
        const uint32_t pixelBits = ScaledTextureBitsPerPixel(source.raw_tex_metadata);
        const uint32_t transferBits = 4u << source.siz;
        const uint64_t bits = (uint64_t(lrs) + 1) * transferBits;
        const uint64_t rowBits = uint64_t(width) * pixelBits;
        const uint32_t height = rowBits != 0 && bits % rowBits == 0
            ? static_cast<uint32_t>(bits / rowBits) : 0;
        LoadScaledTexture(tile, uls * transferBits / pixelBits, ult, width, height, true);
        return;
    }
    const auto byteCountForTexels = [](uint64_t texels, uint8_t siz) -> uint32_t {
        uint64_t bytes = 0;
        switch (siz) {
            case G_IM_SIZ_4b:
                bytes = (texels + 1u) / 2u;
                break;
            case G_IM_SIZ_8b:
                bytes = texels;
                break;
            case G_IM_SIZ_16b:
                bytes = texels * 2u;
                break;
            case G_IM_SIZ_32b:
                bytes = texels * 4u;
                break;
            default:
                break;
        }
        return static_cast<uint32_t>(std::min<uint64_t>(bytes, UINT32_MAX));
    };
    const uint32_t orig_size_bytes = byteCountForTexels(static_cast<uint64_t>(lrs) + 1u,
                                                        mRdp->texture_to_load.siz);
    uint32_t size_bytes = orig_size_bytes;
    if (mRdp->texture_to_load.raw_tex_metadata.h_byte_scale != 1 ||
        mRdp->texture_to_load.raw_tex_metadata.v_pixel_scale != 1) {
        size_bytes *= mRdp->texture_to_load.raw_tex_metadata.h_byte_scale;
        size_bytes *= mRdp->texture_to_load.raw_tex_metadata.v_pixel_scale;
    }
    const uint16_t tmemIndex = mRdp->texture_tile[tile].tmem_index;
    LoadedTexture loaded{};
    loaded.orig_size_bytes = orig_size_bytes;
    loaded.size_bytes = size_bytes;

    // G_LOADBLOCK's SETTIMG size is a transfer size, not necessarily the
    // texture's logical size. I4/I8 macros deliberately transfer as 16-bit.
    // When a render tile for the same TMEM slot already describes this image,
    // use its logical line layout for DRAM row stepping and texture import.
    uint32_t logical_line_bytes = 0;
    for (uint32_t renderTile = 0; renderTile < 8; ++renderTile) {
        const auto& candidateTile = mRdp->texture_tile[renderTile];
        if (renderTile == tile || candidateTile.tmem_index != tmemIndex ||
            candidateTile.line_size_bytes == 0 || mRdp->texture_to_load.width <= 1) {
            continue;
        }
        uint32_t candidateTmemLineBytes = candidateTile.line_size_bytes;
        if (candidateTile.siz == G_IM_SIZ_32b) {
            candidateTmemLineBytes *= 2u;
        }

        // LOADBLOCK's DXT encodes the number of 64-bit TMEM words per source row, which is
        // authoritative when the transfer size differs from the logical render size (I4 data
        // transferred as 16-bit words): the descriptor width can then describe only the visible
        // region while DXT still preserves the real source stride.
        if (dxt != 0 && mRdp->texture_to_load.siz != candidateTile.siz) {
            constexpr uint32_t kDxtOne = 1u << 11;
            const uint32_t dxtWordsPerLine = (kDxtOne + dxt - 1u) / dxt;
            const uint32_t dxtLineBytes = dxtWordsPerLine * 8u;
            if (dxtLineBytes == candidateTmemLineBytes &&
                dxtLineBytes <= size_bytes && size_bytes % dxtLineBytes == 0) {
                logical_line_bytes = dxtLineBytes;
                break;
            }
        }

        // SETTIMG may deliberately use a wider transfer format than the render tile (I4 through a
        // 16-bit LOADBLOCK). The tile's line field describes padded TMEM storage while DRAM rows
        // stay tightly packed, so derive the source stride from the image width and accept it only
        // when its 8-byte-aligned size matches the configured tile line.
        const uint32_t candidateSourceLineBytes =
            byteCountForTexels(mRdp->texture_to_load.width, candidateTile.siz);
        const uint32_t candidateAlignedTmemLineBytes =
            (candidateSourceLineBytes + 7u) & ~7u;
        if (candidateSourceLineBytes != 0 &&
            candidateTmemLineBytes == candidateAlignedTmemLineBytes &&
            candidateSourceLineBytes <= size_bytes &&
            size_bytes % candidateSourceLineBytes == 0) {
            logical_line_bytes = candidateSourceLineBytes;
            break;
        }
    }

    // Fall back to the SETTIMG transfer layout when no logical render tile is
    // available yet. Width 1 is the standard block-load sentinel and means the
    // entire transfer is one contiguous line until SETTILE supplies the layout.
    uint32_t source_line_bytes = logical_line_bytes != 0 ? logical_line_bytes : size_bytes;
    if (mRdp->texture_to_load.width > 1) {
        const uint32_t transferLineBytes =
            byteCountForTexels(mRdp->texture_to_load.width, mRdp->texture_to_load.siz);
        if (logical_line_bytes == 0 && transferLineBytes > 0 &&
            transferLineBytes <= size_bytes && size_bytes % transferLineBytes == 0) {
            source_line_bytes = transferLineBytes;
        }
    }

    // HD pack whole-image replacements (e.g. 4x portraits) carry their real pixel
    // dimensions in raw_tex_metadata. The stock SETTIMG width describes the logical
    // (native) row, so the source stride must be multiplied by the horizontal upscale
    // to step through the replacement's rows correctly.
    const float h_byte_scale = mRdp->texture_to_load.raw_tex_metadata.h_byte_scale;
    const float v_pixel_scale = mRdp->texture_to_load.raw_tex_metadata.v_pixel_scale;
    if (h_byte_scale != 1.0f || v_pixel_scale != 1.0f) {
        source_line_bytes = static_cast<uint32_t>(source_line_bytes * h_byte_scale);
    }

    loaded.line_size_bytes = source_line_bytes;
    loaded.full_image_line_size_bytes = source_line_bytes;
    // assert(size_bytes <= 4096 && "bug: too big texture");
    loaded.tex_flags = mRdp->texture_to_load.tex_flags;
    loaded.raw_tex_metadata = mRdp->texture_to_load.raw_tex_metadata;
    // For upscaled replacements, the LOADBLOCK ult/uls are still in native texels;
    // convert them to replacement texels before computing the source address.
    const size_t rowOffset = static_cast<size_t>(ult) * static_cast<size_t>(v_pixel_scale) * source_line_bytes;
    const size_t columnOffset =
        static_cast<size_t>(byteCountForTexels(uls, mRdp->texture_to_load.siz) * h_byte_scale);
    loaded.addr = mRdp->texture_to_load.addr != nullptr
                      ? mRdp->texture_to_load.addr + rowOffset + columnOffset
                      : nullptr;
    // Per-tile atlas override: keyed on the band's byte offset within its containing registered
    // buffer, which the row/column math above has just baked into loaded.addr (the tile's uls/ult
    // are normally 0 for per-glyph loads). Width 1 is the contiguous-line sentinel and carries no
    // reliable band geometry, so those loads never redirect.
    bool atlasOverridden = false;
    {
        const uint32_t bandWidth = mRdp->texture_to_load.width;
        const uint32_t bandLineBytes = byteCountForTexels(bandWidth, mRdp->texture_to_load.siz);
        if (bandWidth > 1 && bandLineBytes != 0 && orig_size_bytes % bandLineBytes == 0) {
            atlasOverridden = GdxTryAtlasTileOverride(mRdp, loaded, tile, bandWidth,
                                                      orig_size_bytes / bandLineBytes, true);
        }
    }
    // fprintf(stderr, "GfxDpLoadBlock: line_size = 0x%x; orig = 0x%x; bpp=%d; lrs=%d\n", size_bytes,
    // orig_size_bytes,
    //         mRdp->texture_to_load.siz, lrs);

    const std::string_view texPath =
        mRdp->texture_to_load.raw_tex_metadata.resource != nullptr
            ? GetBaseTexturePath(mRdp->texture_to_load.raw_tex_metadata.resource->GetInitData()->Path)
            : std::string_view{};
    auto maskedTextureIter = mMaskedTextures.find(texPath);
    if (maskedTextureIter != mMaskedTextures.end()) {
        loaded.masked = true;
        loaded.blended = maskedTextureIter->second.replacementData != nullptr;
    } else {
        loaded.masked = false;
        loaded.blended = false;
    }

    // Mirror the block transfer into emulated TMEM so import can decode purely from
    // tile-descriptor state. A straight byte copy is correct: the block load's DXT interleave
    // cancels between a linear write at load time and a linear read at import time. An
    // atlas-redirected slot is skipped: its texels come from the pack OTEX, not DRAM, and the
    // resource != nullptr metadata already keeps the importers off the TMEM decode branch.
    if (loaded.addr != nullptr && !atlasOverridden) {
        const uint32_t tmemByteOffset = static_cast<uint32_t>(tmemIndex) * 8u;
        if (tmemByteOffset < sizeof(mRdp->tmem)) {
            uint32_t copyBytes = std::min<uint32_t>(size_bytes, sizeof(mRdp->tmem) - tmemByteOffset);
            const size_t readable = TmemSourceReadableLimit(loaded.addr);
            if (readable < copyBytes) {
                copyBytes = static_cast<uint32_t>(readable);
            }
            if (copyBytes != 0) {
                memcpy(mRdp->tmem + tmemByteOffset, loaded.addr, copyBytes);
                mRdp->tmem_generation++;
            }
        }
    }

    /* One line per distinct (tmem, siz, lrs, dxt): if a decode's slot sizes disagree with what
       this LOADBLOCK stored, the slot was replaced between load and import. */
    {
        static const bool sDiagFontMachineC = std::getenv("GDX_DIAG_FONT_MACHINE") != nullptr;
        if (sDiagFontMachineC) {
            const uint64_t comboKey = (static_cast<uint64_t>(tmemIndex) << 40) |
                                      (static_cast<uint64_t>(mRdp->texture_to_load.siz) << 32) |
                                      (static_cast<uint64_t>(lrs) << 12) | dxt;
            static uint64_t sSeenCombos[96] = {};
            static int sSeenComboCount = 0;
            bool seen = false;
            for (int s = 0; s < sSeenComboCount; s++) {
                if (sSeenCombos[s] == comboKey) {
                    seen = true;
                    break;
                }
            }
            if (!seen && sSeenComboCount < 96) {
                sSeenCombos[sSeenComboCount++] = comboKey;
                gdx_dbg_logf("[fontmach] loadblock tile=%u tmem=0x%X siz=%u w=%u lrs=%u dxt=%u "
                             "sizeB=%u lineB=%u addr=%p\n",
                             tile, tmemIndex, mRdp->texture_to_load.siz, mRdp->texture_to_load.width,
                             lrs, dxt, loaded.size_bytes, loaded.line_size_bytes,
                             static_cast<const void*>(loaded.addr));
            }
        }
    }

    StoreLoadedTexture(tmemIndex, loaded);
    mRdp->textures_changed[0] = true;
    mRdp->textures_changed[1] = true;
}

void Interpreter::GfxDpLoadTile(uint8_t tile, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t lrt) {
    SUPPORT_CHECK(tile == G_TX_LOADTILE);

    if (IsScaledLinearTexture(mRdp->texture_to_load.raw_tex_metadata)) {
        LoadScaledTexture(tile, uls >> 2, ult >> 2, lrs >= uls ? ((lrs - uls) >> 2) + 1 : 0,
                        lrt >= ult ? ((lrt - ult) >> 2) + 1 : 0, false);
        mRdp->texture_tile[tile].uls = uls;
        mRdp->texture_tile[tile].ult = ult;
        mRdp->texture_tile[tile].lrs = lrs;
        mRdp->texture_tile[tile].lrt = lrt;
        return;
    }

    uint32_t word_size_shift = 0;
    switch (mRdp->texture_to_load.siz) {
        case G_IM_SIZ_4b:
            word_size_shift = 0;
            break;
        case G_IM_SIZ_8b:
            word_size_shift = 0;
            break;
        case G_IM_SIZ_16b:
            word_size_shift = 1;
            break;
        case G_IM_SIZ_32b:
            word_size_shift = 2;
            break;
    }

    uint32_t offset_x = uls >> G_TEXTURE_IMAGE_FRAC;
    uint32_t offset_y = ult >> G_TEXTURE_IMAGE_FRAC;
    uint32_t tile_width = ((lrs - uls) >> G_TEXTURE_IMAGE_FRAC) + 1;
    uint32_t tile_height = ((lrt - ult) >> G_TEXTURE_IMAGE_FRAC) + 1;
    uint32_t full_image_width = mRdp->texture_to_load.width;

    uint32_t offset_x_in_bytes = offset_x << word_size_shift;
    uint32_t tile_line_size_bytes = tile_width << word_size_shift;
    uint32_t full_image_line_size_bytes = full_image_width << word_size_shift;

    uint32_t orig_size_bytes = tile_line_size_bytes * tile_height;
    uint32_t size_bytes = orig_size_bytes;
    uint32_t start_offset_bytes = full_image_line_size_bytes * offset_y + offset_x_in_bytes;

    float h_byte_scale = mRdp->texture_to_load.raw_tex_metadata.h_byte_scale;
    float v_pixel_scale = mRdp->texture_to_load.raw_tex_metadata.v_pixel_scale;

    if (h_byte_scale != 1 || v_pixel_scale != 1) {
        start_offset_bytes = h_byte_scale * (v_pixel_scale * offset_y * full_image_line_size_bytes + offset_x_in_bytes);
        size_bytes *= h_byte_scale * v_pixel_scale;
        full_image_line_size_bytes *= h_byte_scale;
        tile_line_size_bytes *= h_byte_scale;
    }

    const uint16_t tmemIndex = mRdp->texture_tile[tile].tmem_index;
    LoadedTexture loaded{};
    loaded.orig_size_bytes = orig_size_bytes;
    loaded.size_bytes = size_bytes;
    loaded.full_image_line_size_bytes = full_image_line_size_bytes;
    loaded.line_size_bytes = tile_line_size_bytes;

    //    assert(size_bytes <= 4096 && "bug: too big texture");
    loaded.tex_flags = mRdp->texture_to_load.tex_flags;
    loaded.raw_tex_metadata = mRdp->texture_to_load.raw_tex_metadata;
    loaded.addr = mRdp->texture_to_load.addr != nullptr ? mRdp->texture_to_load.addr + start_offset_bytes : nullptr;
    // Per-tile atlas override: start_offset_bytes is the band's byte offset within its containing
    // registered buffer, now baked into loaded.addr. tile_width/tile_height are the band dims.
    const bool atlasOverridden =
        GdxTryAtlasTileOverride(mRdp, loaded, tile, tile_width, tile_height, false);

    const std::string_view texPath =
        mRdp->texture_to_load.raw_tex_metadata.resource != nullptr
            ? GetBaseTexturePath(mRdp->texture_to_load.raw_tex_metadata.resource->GetInitData()->Path)
            : std::string_view{};
    auto maskedTextureIter = mMaskedTextures.find(texPath);
    if (maskedTextureIter != mMaskedTextures.end()) {
        loaded.masked = true;
        loaded.blended = maskedTextureIter->second.replacementData != nullptr;
    } else {
        loaded.masked = false;
        loaded.blended = false;
    }

    // TMEM emulation: mirror the tile sub-rectangle into emulated TMEM row by
    // row. Destination rows advance at the 8-byte-aligned tile line stride,
    // matching how the hardware packs LoadTile rows. An atlas-redirected slot is
    // skipped: its texels come from the pack OTEX, not DRAM, and the
    // resource != nullptr metadata already keeps the importers off the TMEM
    // decode branch.
    if (loaded.addr != nullptr && !atlasOverridden) {
        const uint32_t tmemByteOffset = static_cast<uint32_t>(tmemIndex) * 8u;
        const uint32_t destStride = (tile_line_size_bytes + 7u) & ~7u;
        if (tmemByteOffset < sizeof(mRdp->tmem) && destStride != 0) {
            size_t readable = TmemSourceReadableLimit(loaded.addr);
            bool wrote = false;
            for (uint32_t y = 0; y < tile_height; y++) {
                const uint32_t destOff = tmemByteOffset + y * destStride;
                const size_t srcOff = static_cast<size_t>(y) * full_image_line_size_bytes;
                if (destOff >= sizeof(mRdp->tmem) || srcOff + tile_line_size_bytes > readable) {
                    break;
                }
                const uint32_t rowBytes =
                    std::min<uint32_t>(tile_line_size_bytes, sizeof(mRdp->tmem) - destOff);
                memcpy(mRdp->tmem + destOff, loaded.addr + srcOff, rowBytes);
                wrote = true;
            }
            if (wrote) {
                mRdp->tmem_generation++;
            }
        }
    }

    StoreLoadedTexture(tmemIndex, loaded);
    mRdp->texture_tile[tile].uls = uls;
    mRdp->texture_tile[tile].ult = ult;
    mRdp->texture_tile[tile].lrs = lrs;
    mRdp->texture_tile[tile].lrt = lrt;

    mRdp->textures_changed[0] = true;
    mRdp->textures_changed[1] = true;
}

/*static uint8_t color_comb_component(uint32_t v) {
    switch (v) {
        case G_CCMUX_TEXEL0:
            return CC_TEXEL0;
        case G_CCMUX_TEXEL1:
            return CC_TEXEL1;
        case G_CCMUX_PRIMITIVE:
            return CC_PRIM;
        case G_CCMUX_SHADE:
            return CC_SHADE;
        case G_CCMUX_ENVIRONMENT:
            return CC_ENV;
        case G_CCMUX_TEXEL0_ALPHA:
            return CC_TEXEL0A;
        case G_CCMUX_LOD_FRACTION:
            return CC_LOD;
        default:
            return CC_0;
    }
}

static inline uint32_t color_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return color_comb_component(a) |
           (color_comb_component(b) << 3) |
           (color_comb_component(c) << 6) |
           (color_comb_component(d) << 9);
}

static void GfxDpSetCombineMode(uint32_t rgb, uint32_t alpha) {
    mRdp->combine_mode = rgb | (alpha << 12);
}*/

void Interpreter::GfxDpSetCombineMode(uint32_t rgb, uint32_t alpha, uint32_t rgb_cyc2, uint32_t alpha_cyc2) {
    mRdp->combine_mode = rgb | (alpha << 16) | ((uint64_t)rgb_cyc2 << 28) | ((uint64_t)alpha_cyc2 << 44);
}

static inline uint32_t color_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return (a & 0xf) | ((b & 0xf) << 4) | ((c & 0x1f) << 8) | ((d & 7) << 13);
}

static inline uint32_t alpha_comb(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return (a & 7) | ((b & 7) << 3) | ((c & 7) << 6) | ((d & 7) << 9);
}

// Sign-extend a 9-bit value (used for G_SETCONVERT K0..K5).
static inline int16_t sign_extend_9(uint32_t v) {
    return (int16_t)((v & 0x100) ? (int32_t)(v | 0xFFFFFE00u) : (int32_t)v);
}

void Interpreter::GfxDpSetGrayscaleColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->grayscale_color.r = r;
    mRdp->grayscale_color.g = g;
    mRdp->grayscale_color.b = b;
    mRdp->grayscale_color.a = a;
}

void Interpreter::GfxDpSetEnvColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->env_color.r = r;
    mRdp->env_color.g = g;
    mRdp->env_color.b = b;
    mRdp->env_color.a = a;
}

void Interpreter::GfxDpSetPrimColor(uint8_t m, uint8_t l, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->prim_lod_fraction = l;
    mRdp->prim_color.r = r;
    mRdp->prim_color.g = g;
    mRdp->prim_color.b = b;
    mRdp->prim_color.a = a;
}

void Interpreter::GfxDpSetFogColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->fog_color.r = r;
    mRdp->fog_color.g = g;
    mRdp->fog_color.b = b;
    mRdp->fog_color.a = a;
}

void Interpreter::GfxDpSetBlendColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    mRdp->blend_color.r = r;
    mRdp->blend_color.g = g;
    mRdp->blend_color.b = b;
    mRdp->blend_color.a = a;
}

void Interpreter::GfxDpSetFillColor(uint32_t packed_color) {
    uint16_t col16 = (uint16_t)packed_color;
    uint32_t r = col16 >> 11;
    uint32_t g = (col16 >> 6) & 0x1f;
    uint32_t b = (col16 >> 1) & 0x1f;
    uint32_t a = col16 & 1;
    mRdp->fill_color.r = SCALE_5_8(r);
    mRdp->fill_color.g = SCALE_5_8(g);
    mRdp->fill_color.b = SCALE_5_8(b);
    mRdp->fill_color.a = a * 255;
}

void Interpreter::GfxDrawRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry) {
    uint32_t saved_other_mode_h = mRdp->other_mode_h;
    uint32_t cycle_type = (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE));

    if (cycle_type == G_CYC_COPY) {
        mRdp->other_mode_h = (mRdp->other_mode_h & ~(3U << G_MDSFT_TEXTFILT)) | G_TF_POINT;
    }

    // U10.2 coordinates
    float ulxf = ulx;
    float ulyf = uly;
    float lrxf = lrx;
    float lryf = lry;

    ulxf = ulxf / (4.0f * HALF_SCREEN_WIDTH(mActiveFrameBuffer)) - 1.0f;
    ulyf = -(ulyf / (4.0f * HALF_SCREEN_HEIGHT(mActiveFrameBuffer))) + 1.0f;
    lrxf = lrxf / (4.0f * HALF_SCREEN_WIDTH(mActiveFrameBuffer)) - 1.0f;
    lryf = -(lryf / (4.0f * HALF_SCREEN_HEIGHT(mActiveFrameBuffer))) + 1.0f;

    {
        const bool fixedAspectFramebuffer =
            mFbActive && mActiveFrameBuffer != mFrameBuffers.end() &&
            (!mActiveFrameBuffer->second.resize || mActiveFrameBuffer->second.forceFixedAspect);
        const float currentAspect = (float)mCurDimensions.width / (float)mCurDimensions.height;
        const float aspectScale = (4.0f / 3.0f) / currentAspect;
        const uint32_t widescreenMode =
            mRsp->extra_geometry_mode &
            (G_EX_WIDESCREEN_STRETCH | G_EX_WIDESCREEN_ANCHOR_LEFT | G_EX_WIDESCREEN_ANCHOR_RIGHT |
             G_EX_WIDESCREEN_DISTRIBUTE);
        // CVar states are latched per frame in StartFrame. Forced-4:3 frames composite into a
        // centred pillarbox, so a stretch or anchor scope that slipped into such a frame would be
        // double-corrected afterwards; the scopes go inert with the frame.
        const bool widescreenFrameActive = !fixedAspectFramebuffer && currentAspect > (4.0f / 3.0f) &&
                                           mWidescreenEnabledCache && !mForceFixedAspectCache;
        // STRETCH follows the 3D Widescreen CVar alone: transitions redraw a captured widescreen
        // frame and must cover the full viewport whenever the live frame renders widescreen.
        // ANCHOR/DISTRIBUTE reposition individual 2D elements and stay behind the WidescreenUI
        // opt-in.
        const bool stretchActive = widescreenFrameActive && (widescreenMode & G_EX_WIDESCREEN_STRETCH) != 0;
        const bool enhancedWidescreenUi = widescreenFrameActive && widescreenMode != 0 && mWidescreenUiCache;

        if (stretchActive) {
            // F-Zero X's fullscreen 2D safe area is x=12..308. Map that explicit draw scope to
            // the true viewport edges; foreground/menu artwork remains on the normal 4:3 path.
            constexpr float kSafeAreaScale = 160.0f / (160.0f - 12.0f);
            ulxf *= kSafeAreaScale;
            lrxf *= kSafeAreaScale;
        } else if (enhancedWidescreenUi && (widescreenMode & G_EX_WIDESCREEN_DISTRIBUTE) != 0) {
            // Place a 2D widget by its native 320-wide centre so it follows a widescreen-spread
            // 3D viewport, but retain the widget's 4:3-corrected width. This is distinct from
            // STRETCH: SELECT MACHINE's cursor must move with its six 3D columns without becoming
            // horizontally distorted.
            const float center = (ulxf + lrxf) * 0.5f;
            const float halfWidth = (lrxf - ulxf) * 0.5f * aspectScale;
            ulxf = center - halfWidth;
            lrxf = center + halfWidth;
        } else {
            ulxf = AdjXForAspectRatio(ulxf);
            lrxf = AdjXForAspectRatio(lrxf);

            if (enhancedWidescreenUi) {
                const bool anchorLeft = (widescreenMode & G_EX_WIDESCREEN_ANCHOR_LEFT) != 0;
                const bool anchorRight = (widescreenMode & G_EX_WIDESCREEN_ANCHOR_RIGHT) != 0;
                if (anchorLeft != anchorRight) {
                    // Clamp anchor travel to a centred band of aspect min(current, HudMaxAspect)
                    // rather than the raw viewport edge, which parks HUD elements at the physical
                    // corners of arbitrarily wide displays. The band edge sits at NDC
                    // uiAspect/currentAspect, so travel is (uiAspect - 4/3)/currentAspect --
                    // exactly (1 - aspectScale), the old formula, for every aspect <=
                    // HudMaxAspect.
                    const float currentAspectForUi =
                        (float)mCurDimensions.width / (float)mCurDimensions.height;
                    const float uiAspect = std::min(currentAspectForUi, mHudMaxAspectCache);
                    float offset = (anchorRight ? 1.0f : -1.0f) *
                                   ((uiAspect - (4.0f / 3.0f)) / currentAspectForUi);
                    if (mRemoveBordersCache) {
                        // HUD elements are authored 12 px inside the overscan-safe frame; with the
                        // frame removed, extend anchor travel by that margin (12/160 in 4:3 NDC)
                        // so they glue to the true corners instead of the retired safe edge.
                        offset += (anchorRight ? 1.0f : -1.0f) * (0.1f / currentAspectForUi);
                    }
                    ulxf += offset;
                    lrxf += offset;
                }
            }
        }
    }

    struct LoadedVertex* ul = &mRsp->loaded_vertices[MAX_VERTICES + 0];
    struct LoadedVertex* ll = &mRsp->loaded_vertices[MAX_VERTICES + 1];
    struct LoadedVertex* lr = &mRsp->loaded_vertices[MAX_VERTICES + 2];
    struct LoadedVertex* ur = &mRsp->loaded_vertices[MAX_VERTICES + 3];

    ul->x = ulxf;
    ul->y = ulyf;
    ul->z = -1.0f;
    ul->w = 1.0f;

    ll->x = ulxf;
    ll->y = lryf;
    ll->z = -1.0f;
    ll->w = 1.0f;

    lr->x = lrxf;
    lr->y = lryf;
    lr->z = -1.0f;
    lr->w = 1.0f;

    ur->x = lrxf;
    ur->y = ulyf;
    ur->z = -1.0f;
    ur->w = 1.0f;

    // The coordinates for texture rectangle shall bypass the viewport setting
    struct XYWidthHeight default_viewport;
    if (!mFbActive) {
        default_viewport = { 0, (int16_t)mNativeDimensions.height, mNativeDimensions.width, mNativeDimensions.height };
    } else {
        default_viewport = { 0, (int16_t)mActiveFrameBuffer->second.orig_height, mActiveFrameBuffer->second.orig_width,
                             mActiveFrameBuffer->second.orig_height };
    }

    struct XYWidthHeight viewport_saved = mRdp->viewport;
    uint32_t geometry_mode_saved = mRsp->geometry_mode;

    AdjustVIewportOrScissor(&default_viewport);

    mRdp->viewport = default_viewport;
    mRdp->viewport_or_scissor_changed = true;
    mRsp->geometry_mode = 0;

    GfxSpTri1(MAX_VERTICES + 0, MAX_VERTICES + 1, MAX_VERTICES + 3, true);
    GfxSpTri1(MAX_VERTICES + 1, MAX_VERTICES + 2, MAX_VERTICES + 3, true);

    mRsp->geometry_mode = geometry_mode_saved;
    mRdp->viewport = viewport_saved;
    mRdp->viewport_or_scissor_changed = true;

    if (cycle_type == G_CYC_COPY) {
        mRdp->other_mode_h = saved_other_mode_h;
    }
}

void Interpreter::GfxDpTextureRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry, uint8_t tile, int16_t uls,
                                        int16_t ult, int16_t dsdx, int16_t dtdy, bool flip) {
    // VR: a screen-space sprite at a fixed pixel position in an eye image sits at a different
    // angle in each eye and is seen double. World passes therefore draw only world-anchored
    // sprites, re-placed for the eye; HUD-bracketed sprites go to the HUD pass; anything else
    // drawn before the HUD marker (e.g. stars) is left out. The HUD pass never draws world sprites.
    if (mVrEyeActive && mVrPass == VrPass::WorldOnly) {
        if (!mVrWorldSpriteActive || !VrPlaceWorldSprite(&ulx, &uly, &lrx, &lry, &uls, &ult, &dsdx, &dtdy)) {
            return;
        }
        gGdxVrWorldSpritesLastRun++;
    } else if (mVrEyeActive && mVrPass == VrPass::HudOnly) {
        if (mVrWorldSpriteActive) {
            return;
        }
        if (mVrInHudSprites) {
            gGdxVrHudSpritesLastRun++;
        }
    }
    // printf("render %d at %d\n", tile, lrx);
    uint64_t saved_combine_mode = mRdp->combine_mode;
    if ((mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE)) == G_CYC_COPY) {
        // Per RDP Command Summary Set Tile's shift s and this dsdx should be set to 4 texels
        // Divide by 4 to get 1, but only when the game followed the 4x convention. F-Zero X emits
        // dsdx=0x0400 (already 1.0 texel/pixel) in copy mode, where dividing would give a 4x zoom,
        // so skip the shift when the magnitude is already at or below 1.0.
        if (dsdx > 0x0400 || dsdx < -0x0400) {
            dsdx >>= 2;
        }

        // Color combiner is turned off in copy mode
        GfxDpSetCombineMode(color_comb(0, 0, 0, G_CCMUX_TEXEL0), alpha_comb(0, 0, 0, G_ACMUX_TEXEL0), 0, 0);

        // Per documentation one extra pixel is added in this modes to each edge
        lrx += 1 << 2;
        lry += 1 << 2;
    }

    // uls and ult are S10.5
    // dsdx and dtdy are S5.10
    // lrx, lry, ulx, uly are U10.2
    // lrs, lrt are S10.5
    if (flip) {
        dsdx = -dsdx;
        dtdy = -dtdy;
    }
    int16_t width = !flip ? lrx - ulx : lry - uly;
    int16_t height = !flip ? lry - uly : lrx - ulx;
    float lrs = ((uls << 7) + dsdx * width) >> 7;
    float lrt = ((ult << 7) + dtdy * height) >> 7;

    LoadedVertex* ul = &mRsp->loaded_vertices[MAX_VERTICES + 0];
    LoadedVertex* ll = &mRsp->loaded_vertices[MAX_VERTICES + 1];
    LoadedVertex* lr = &mRsp->loaded_vertices[MAX_VERTICES + 2];
    LoadedVertex* ur = &mRsp->loaded_vertices[MAX_VERTICES + 3];
    ul->u = uls;
    ul->v = ult;
    lr->u = lrs;
    lr->v = lrt;
    if (!flip) {
        ll->u = uls;
        ll->v = lrt;
        ur->u = lrs;
        ur->v = ult;
    } else {
        ll->u = lrs;
        ll->v = ult;
        ur->u = uls;
        ur->v = lrt;
    }

    uint8_t saved_tile = mRdp->first_tile_index;
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = tile;

    GfxDrawRectangle(ulx, uly, lrx, lry);
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = saved_tile;
    mRdp->combine_mode = saved_combine_mode;
}

void Interpreter::GfxDpImageRectangle(int32_t tile, int32_t w, int32_t h, int32_t ulx, int32_t uly, int16_t uls,
                                      int16_t ult, int32_t lrx, int32_t lry, int16_t lrs, int16_t lrt) {

    LoadedVertex* ul = &mRsp->loaded_vertices[MAX_VERTICES + 0];
    LoadedVertex* ll = &mRsp->loaded_vertices[MAX_VERTICES + 1];
    LoadedVertex* lr = &mRsp->loaded_vertices[MAX_VERTICES + 2];
    LoadedVertex* ur = &mRsp->loaded_vertices[MAX_VERTICES + 3];
    ul->u = uls * 32;
    ul->v = ult * 32;
    lr->u = lrs * 32;
    lr->v = lrt * 32;
    ll->u = uls * 32;
    ll->v = lrt * 32;
    ur->u = lrs * 32;
    ur->v = ult * 32;

    // ensure we have the correct texture size, format and starting position
    mRdp->texture_tile[tile].siz = G_IM_SIZ_8b;
    mRdp->texture_tile[tile].fmt = G_IM_FMT_RGBA;
    mRdp->texture_tile[tile].cms = 0;
    mRdp->texture_tile[tile].cmt = 0;
    mRdp->texture_tile[tile].shifts = 0;
    mRdp->texture_tile[tile].shiftt = 0;
    mRdp->texture_tile[tile].uls = 0 * 4;
    mRdp->texture_tile[tile].ult = 0 * 4;
    mRdp->texture_tile[tile].lrs = w * 4;
    mRdp->texture_tile[tile].lrt = h * 4;
    mRdp->texture_tile[tile].line_size_bytes = w << (mRdp->texture_tile[tile].siz >> 1);

    auto& loadtex = mRdp->loaded_texture[mRdp->texture_tile[tile].tmem_index];
    loadtex.full_image_line_size_bytes = loadtex.line_size_bytes = mRdp->texture_tile[tile].line_size_bytes;
    loadtex.size_bytes = loadtex.orig_size_bytes = loadtex.line_size_bytes * h;

    uint8_t saved_tile = mRdp->first_tile_index;
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = tile;

    GfxDrawRectangle(ulx, uly, lrx, lry);
    if (saved_tile != tile) {
        mRdp->textures_changed[0] = true;
        mRdp->textures_changed[1] = true;
    }
    mRdp->first_tile_index = saved_tile;
}

void Interpreter::GfxDpFillRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry) {
    if (mRdp->color_image_address == mRdp->z_buf_address) {
        // Fullscreen Z clears are redundant — already done by glClear at frame start.
        bool isFullScreen = (ulx <= 0 && uly <= 0 && lrx >= (int32_t)(mNativeDimensions.width - 1) * 4 &&
                             lry >= (int32_t)(mNativeDimensions.height - 1) * 4);
        if (isFullScreen) {
            return;
        }

        // Partial depth clear (e.g. HUD model regions): clear the actual depth buffer
        // via a scissored depth clear instead of drawing a colored rect to the color buffer.
        Flush();

        // Convert U10.2 coords to pixel coords and add +1 pixel for fill mode
        int32_t expanded_lrx = lrx + (1 << 2);
        int32_t expanded_lry = lry + (1 << 2);
        float x = ulx / 4.0f;
        float y = expanded_lry / 4.0f;
        float w = (expanded_lrx - ulx) / 4.0f;
        float h = (expanded_lry - uly) / 4.0f;

        struct XYWidthHeight area;
        area.x = (int16_t)x;
        area.y = (int16_t)y;
        area.width = (uint32_t)w;
        area.height = (uint32_t)h;
        AdjustVIewportOrScissor(&area);

        mRapi->ClearDepthRegion(area.x, area.y, area.width, area.height);
        return;
    }
    uint32_t mode = (mRdp->other_mode_h & (3U << G_MDSFT_CYCLETYPE));

    // Hardware draws nothing for an inverted rect pair: the span walker yields a
    // negative scanline count. F-Zero X hits this when a scrolled rect's 10-bit
    // coordinate wraps negative (records_entry.c via ending.c) — do not "fix" this
    // into a clamp, which would rasterise a full-height stripe the hardware never
    // drew. Strict '>': an equal pair is a legitimate 1-pixel rect. Placed before
    // the +1 expansion below so the comparison uses the values the RDP compares.
    if (ulx > lrx || uly > lry) {
        return;
    }

    // Expand fullscreen fill rects to cover widescreen viewports.
    // Without this, screen clears and fades only cover the native 4:3 area.
    if (ulx == 0 && uly == 0) {
        bool isFullScreen = (lrx == ((int32_t)(mNativeDimensions.width - 1) * 4) &&
                             lry == ((int32_t)(mNativeDimensions.height - 1) * 4));
        if (isFullScreen) {
            // The fixed +-1024/2048 extents cover NDC ~+-2.2 after hor+ compression, which stops
            // covering the frame above aspect ~2.93 (unfaded side bands at 32:9). Derive the
            // horizontal extent from the live aspect instead, floored at the old constants:
            // anything wider than the screen is scissored identically, so 16:9-and-below output
            // is unchanged.
            const float fsAspect = (float)mCurDimensions.width / (float)mCurDimensions.height;
            const float fsK = (4.0f / 3.0f) / std::max(fsAspect, 4.0f / 3.0f);
            int32_t ex = (int32_t)(160.0f * (1.0f / std::max(fsK, 0.05f) + 1.0f) * 4.0f) + 64;
            if (ex < 2048) {
                ex = 2048;
            }
            ulx = -ex;
            uly = -1024;
            lrx = ex;
            lry = 2048;
        }
    }

    if (mode == G_CYC_COPY || mode == G_CYC_FILL) {
        // Per documentation one extra pixel is added in this modes to each edge
        lrx += 1 << 2;
        lry += 1 << 2;
    }

    for (int i = MAX_VERTICES; i < MAX_VERTICES + 4; i++) {
        LoadedVertex* v = &mRsp->loaded_vertices[i];
        v->color = mRdp->fill_color;
    }

    uint64_t saved_combine_mode = mRdp->combine_mode;

    if (mode == G_CYC_FILL) {
        GfxDpSetCombineMode(color_comb(0, 0, 0, G_CCMUX_SHADE), alpha_comb(0, 0, 0, G_ACMUX_SHADE), 0, 0);
    }

    GfxDrawRectangle(ulx, uly, lrx, lry);
    mRdp->combine_mode = saved_combine_mode;
}

void Interpreter::GfxDpSetZImage(void* zBufAddr) {
    mRdp->z_buf_address = zBufAddr;
}

void Interpreter::GfxDpSetColorImage(uint32_t format, uint32_t size, uint32_t width, void* address) {
    mRdp->color_image_address = address;
}

void Interpreter::GfxSpSetOtherMode(uint32_t shift, uint32_t num_bits, uint64_t mode) {
    uint64_t mask = (((uint64_t)1 << num_bits) - 1) << shift;
    uint64_t om = mRdp->other_mode_l | ((uint64_t)mRdp->other_mode_h << 32);
    om = (om & ~mask) | mode;
    mRdp->other_mode_l = (uint32_t)om;
    mRdp->other_mode_h = (uint32_t)(om >> 32);
}

void Interpreter::GfxDpSetOtherMode(uint32_t h, uint32_t l) {
    mRdp->other_mode_h = h;
    mRdp->other_mode_l = l;
}

void Interpreter::Gfxs2dexBgCopy(F3DuObjBg* bg) {
    /*
    bg->b.imageX = 0;
    bg->b.imageW = width * 4;
    bg->b.frameX = frameX * 4;
    bg->b.imageY = 0;
    bg->b.imageH = height * 4;
    bg->b.frameY = frameY * 4;
    bg->b.imagePtr = source;
    bg->b.imageLoad = G_BGLT_LOADTILE;
    bg->b.imageFmt = fmt;
    bg->b.imageSiz = siz;
    bg->b.imagePal = 0;
    bg->b.imageFlip = 0;
    */

    uintptr_t data = (uintptr_t)bg->b.imagePtr;

    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    if ((bool)gfx_check_image_signature((char*)data)) {
        std::shared_ptr<Fast::Texture> tex = std::static_pointer_cast<Fast::Texture>(
            Ship::Context::GetInstance()->GetResourceManager()->LoadResourceProcess((char*)data));
        texFlags = tex->Flags;
        rawTexMetadata.width = tex->Width;
        rawTexMetadata.height = tex->Height;
        rawTexMetadata.h_byte_scale = tex->HByteScale;
        rawTexMetadata.v_pixel_scale = tex->VPixelScale;
        rawTexMetadata.type = tex->Type;
        rawTexMetadata.resource = tex;
        data = (uintptr_t) reinterpret_cast<char*>(tex->ImageData);
    }

    s16 dsdx = 4 << 10;
    s16 uls = bg->b.imageX << 3;
    // Flip flag only flips horizontally
    if (bg->b.imageFlip == G_BG_FLAG_FLIPS) {
        dsdx = -dsdx;
        uls = (bg->b.imageW - bg->b.imageX) << 3;
    }

    SUPPORT_CHECK(bg->b.imageSiz == G_IM_SIZ_16b);
    GfxDpSetTextureImage(G_IM_FMT_RGBA, G_IM_SIZ_16b, 0, nullptr, texFlags, rawTexMetadata, (void*)data);
    GfxDpSetTile(G_IM_FMT_RGBA, G_IM_SIZ_16b, 0, 0, G_TX_LOADTILE, 0, 0, 0, 0, 0, 0, 0);
    GfxDpLoadBlock(G_TX_LOADTILE, 0, 0, (bg->b.imageW * bg->b.imageH >> 4) - 1, 0);
    GfxDpSetTile(bg->b.imageFmt, G_IM_SIZ_16b, bg->b.imageW >> 4, 0, G_TX_RENDERTILE, bg->b.imagePal, 0, 0, 0, 0, 0, 0);
    GfxDpSetTileSize(G_TX_RENDERTILE, 0, 0, bg->b.imageW, bg->b.imageH);
    GfxDpTextureRectangle(bg->b.frameX, bg->b.frameY, bg->b.frameX + bg->b.imageW - 4, bg->b.frameY + bg->b.imageH - 4,
                          G_TX_RENDERTILE, uls, bg->b.imageY << 3, dsdx, 1 << 10, false);
}

void Interpreter::Gfxs2dexBg1cyc(F3DuObjBg* bg) {
    uintptr_t data = (uintptr_t)bg->b.imagePtr;

    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    if ((bool)gfx_check_image_signature((char*)data)) {
        std::shared_ptr<Fast::Texture> tex = std::static_pointer_cast<Fast::Texture>(
            Ship::Context::GetInstance()->GetResourceManager()->LoadResourceProcess((char*)data));
        texFlags = tex->Flags;
        rawTexMetadata.width = tex->Width;
        rawTexMetadata.height = tex->Height;
        rawTexMetadata.h_byte_scale = tex->HByteScale;
        rawTexMetadata.v_pixel_scale = tex->VPixelScale;
        rawTexMetadata.type = tex->Type;
        rawTexMetadata.resource = tex;
        data = (uintptr_t) reinterpret_cast<char*>(tex->ImageData);
    }

    // TODO: Implement bg scaling correctly
    s16 uls = bg->b.imageX >> 2;
    s16 lrs = bg->b.imageW >> 2;

    s16 dsdxRect = 1 << 10;
    s16 ulsRect = bg->b.imageX << 3;
    // Flip flag only flips horizontally
    if (bg->b.imageFlip == G_BG_FLAG_FLIPS) {
        dsdxRect = -dsdxRect;
        ulsRect = (bg->b.imageW - bg->b.imageX) << 3;
    }

    GfxDpSetTextureImage(bg->b.imageFmt, bg->b.imageSiz, bg->b.imageW >> 2, nullptr, texFlags, rawTexMetadata,
                         (void*)data);
    GfxDpSetTile(bg->b.imageFmt, bg->b.imageSiz, 0, 0, G_TX_LOADTILE, 0, 0, 0, 0, 0, 0, 0);
    GfxDpLoadBlock(G_TX_LOADTILE, 0, 0, (bg->b.imageW * bg->b.imageH >> 4) - 1, 0);
    GfxDpSetTile(bg->b.imageFmt, bg->b.imageSiz, (((lrs - uls) * bg->b.imageSiz) + 7) >> 3, 0, G_TX_RENDERTILE,
                 bg->b.imagePal, 0, 0, 0, 0, 0, 0);
    GfxDpSetTileSize(G_TX_RENDERTILE, 0, 0, bg->b.imageW, bg->b.imageH);

    GfxDpTextureRectangle(bg->b.frameX, bg->b.frameY, bg->b.frameW, bg->b.frameH, G_TX_RENDERTILE, ulsRect,
                          bg->b.imageY << 3, dsdxRect, 1 << 10, false);
}

void Interpreter::Gfxs2dexRecyCopy(F3DuObjSprite* spr) {
    s16 dsdx = 4 << 10;
    [[maybe_unused]] s16 uls = spr->s.objX << 3;
    // Flip flag only flips horizontally
    if (spr->s.imageFlags == G_BG_FLAG_FLIPS) {
        dsdx = -dsdx;
        uls = (spr->s.imageW - spr->s.objX) << 3;
    }

    int realX = spr->s.objX >> 2;
    int realY = spr->s.objY >> 2;
    int realW = (((spr->s.imageW)) >> 5);
    int realH = (((spr->s.imageH)) >> 5);
    float realSW = spr->s.scaleW / 1024.0f;
    float realSH = spr->s.scaleH / 1024.0f;

    int testX = (realX + (realW / realSW));
    int testY = (realY + (realH / realSH));

    GfxDpTextureRectangle(realX << 2, realY << 2, testX << 2, testY << 2, G_TX_RENDERTILE,
                          (s32)mRdp->texture_tile[0].uls << 3, (s32)mRdp->texture_tile[0].ult << 3,
                          (float)(1 << 10) * realSW, (float)(1 << 10) * realSH, false);
}

void* Interpreter::SegAddr(uintptr_t w1) {
    // Segmented?
    if (w1 & 1) {
        uint32_t segNum = (uint32_t)(w1 >> 24);

        uint32_t offset = w1 & 0x00FFFFFE;

        if (mSegmentPointers[segNum] != 0) {
            return (void*)(mSegmentPointers[segNum] + offset);
        } else {
            return (void*)w1;
        }
    } else {
        return (void*)w1;
    }
}

#define C0(pos, width) ((cmd->words.w0 >> (pos)) & ((1U << width) - 1))
#define C1(pos, width) ((cmd->words.w1 >> (pos)) & ((1U << width) - 1))

void GfxExecStack::start(F3DGfx* dlist) {
    while (!cmd_stack.empty())
        cmd_stack.pop();
    gfx_path.clear();
    cmd_stack.push(dlist);
    disp_stack.clear();
}

void GfxExecStack::stop() {
    while (!cmd_stack.empty())
        cmd_stack.pop();
    gfx_path.clear();
}

F3DGfx*& GfxExecStack::currCmd() {
    return cmd_stack.top();
}

void GfxExecStack::openDisp(const char* file, int line) {
    disp_stack.push_back({ file, line });
}
void GfxExecStack::closeDisp() {
    disp_stack.pop_back();
}
const std::vector<GfxExecStack::CodeDisp>& GfxExecStack::getDisp() const {
    return disp_stack;
}

void GfxExecStack::branch(F3DGfx* caller) {
    F3DGfx* old = cmd_stack.top();
    cmd_stack.pop();
    cmd_stack.push(nullptr);
    cmd_stack.push(old);

    gfx_path.push_back(caller);
}

void GfxExecStack::call(F3DGfx* caller, F3DGfx* callee) {
    cmd_stack.push(callee);
    gfx_path.push_back(caller);
}

F3DGfx* GfxExecStack::ret() {
    F3DGfx* cmd = cmd_stack.top();

    cmd_stack.pop();
    if (!gfx_path.empty()) {
        gfx_path.pop_back();
    }

    while (cmd_stack.size() > 0 && cmd_stack.top() == nullptr) {
        cmd_stack.pop();
        if (!gfx_path.empty()) {
            gfx_path.pop_back();
        }
    }
    return cmd;
}

void gfx_set_framebuffer(int fb, float noise_scale);
void gfx_reset_framebuffer();
void gfx_copy_framebuffer(int fb_dst_id, int fb_src_id, bool copyOnce, bool* hasCopiedPtr);

// The main type of the handler function. These function will take a pointer to a pointer to a Gfx. It needs to be a
// double pointer because we sometimes need to increment and decrement the underlying pointer Returns false if the
// current opcode should be incremented after the handler ends.
typedef bool (*GfxOpcodeHandlerFunc)(F3DGfx** gfx);

bool gfx_load_ucode_handler_f3dex2(F3DGfx** cmd) {
    Interpreter* gfx = mInstance.lock().get();
    gfx->mRsp->fog_mul = 0;
    gfx->mRsp->fog_offset = 0;
    return false;
}

bool gfx_cull_dl_handler_f3dex2(F3DGfx** cmd) {
    // TODO:
    return false;
}

bool gfx_marker_handler_otr(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    (*cmd0)++;
    F3DGfx* cmd = (*cmd0);
    gfx->mMarkerOn = true;
    return false;
}

bool gfx_invalidate_tex_cache_handler_f3dex2(F3DGfx** cmd) {
    Interpreter* gfx = mInstance.lock().get();
    const uintptr_t texAddr = (*cmd)->words.w1;

    if (texAddr == 0) {
        gfx->TextureCacheClear();
    } else {
        gfx->TextureCacheDelete((const uint8_t*)texAddr);
    }
    return false;
}

bool gfx_noop_handler_f3dex2(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    const char* filename = (const char*)(cmd)->words.w1;
    uint32_t p = C0(16, 8);
    uint32_t l = C0(0, 16);
    if (p == Interpreter::kVrHudMarker) {
        mInstance.lock()->OnVrHudMarker();
    } else if (p >= Interpreter::kVrHudSpritesBegin && p <= Interpreter::kVrWorldSpriteEnd) {
        mInstance.lock()->OnVrSpriteMarker(p, l);
    } else if (p == 7) {
        g_exec_stack.openDisp(filename, l);
    } else if (p == 8) {
        if (g_exec_stack.disp_stack.size() == 0) {
            SPDLOG_WARN("CLOSE_DISPS without matching open {}:{}", p, l);
        } else {
            g_exec_stack.closeDisp();
        }
    }
    return false;
}

bool gfx_mtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    uintptr_t mtxAddr = cmd->words.w1;

    gfx->GfxSpMatrix(C0(0, 8) ^ F3DEX2_G_MTX_PUSH, (const int32_t*)gfx->SegAddr(mtxAddr));
    return false;
}
// Seems to be the same for all other non F3DEX2 microcodes...
bool gfx_mtx_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    uintptr_t mtxAddr = cmd->words.w1;

    gfx->GfxSpMatrix(C0(16, 8), (const int32_t*)gfx->SegAddr(cmd->words.w1));
    return false;
}

bool gfx_mtx_otr_filepath_handler_custom_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    const char* fileName = (const char*)cmd->words.w1;
    const int32_t* mtx = (const int32_t*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(
        (const char*)fileName);

    if (mtx != NULL) {
        gfx->GfxSpMatrix(C0(0, 8) ^ F3DEX2_G_MTX_PUSH, mtx);
    }

    return false;
}

bool gfx_mtx_otr_filepath_handler_custom_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    const char* fileName = (const char*)cmd->words.w1;
    const int32_t* mtx = (const int32_t*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(
        (const char*)fileName);

    if (mtx != NULL) {
        gfx->GfxSpMatrix(C0(16, 8), mtx);
    }

    return false;
}

bool gfx_mtx_otr_filepath_handler_custom(F3DGfx** cmd0) {
    if (ucode_handler_index == ucode_f3dex2) {
        return gfx_mtx_otr_filepath_handler_custom_f3dex2(cmd0);
    } else {
        return gfx_mtx_otr_filepath_handler_custom_f3d(cmd0);
    }
}

bool gfx_mtx_otr_handler_custom_f3dex2(F3DGfx** cmd0) {
    (*cmd0)++;
    F3DGfx* cmd = *cmd0;

    const uint64_t hash = ((uint64_t)cmd->words.w0 << 32) + cmd->words.w1;
    const int32_t* mtx =
        (const int32_t*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash);

    if (mtx != NULL) {
        Interpreter* gfx = mInstance.lock().get();
        cmd--;
        gfx->GfxSpMatrix(C0(0, 8) ^ F3DEX2_G_MTX_PUSH, mtx);
        cmd++;
    }

    return false;
}

bool gfx_mtx_otr_handler_custom_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    (*cmd0)++;
    F3DGfx* cmd = *cmd0;

    const uint64_t hash = ((uint64_t)cmd->words.w0 << 32) + cmd->words.w1;
    const int32_t* mtx =
        (const int32_t*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash);
    if (mtx != nullptr) {
        cmd--;
        gfx->GfxSpMatrix(C0(16, 8), mtx);
        cmd++;
    }
    return false;
}

bool gfx_mtx_otr_handler_custom(F3DGfx** cmd0) {
    if (ucode_handler_index == ucode_f3dex2) {
        return gfx_mtx_otr_handler_custom_f3dex2(cmd0);
    } else {
        return gfx_mtx_otr_handler_custom_f3d(cmd0);
    }
}

bool gfx_pop_mtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpPopMatrix((uint32_t)(cmd->words.w1 / 64));

    return false;
}

bool gfx_pop_mtx_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpPopMatrix(1);

    return false;
}

bool gfx_movemem_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovememF3dex2(C0(0, 8), C0(8, 8) * 8, gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_dma_io_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    const bool write = C0(23, 1) != 0;
    const uint16_t dmem = static_cast<uint16_t>(C0(13, 10) * 8);
    const size_t size = static_cast<size_t>(C0(0, 12)) + 1;
    gfx->GfxSpDmaIo(write, dmem, gfx->SegAddr(cmd->words.w1), size);
    return false;
}

bool gfx_movemem_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovememF3d(C0(16, 8), 0, gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_movemem_handler_otr(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    const uint8_t index = C1(24, 8);
    const uint8_t offset = C1(16, 8);
    const uint8_t hasOffset = C1(8, 8);

    (*cmd0)++;

    const uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

    if (ucode_handler_index == ucode_f3dex2) {
        gfx->GfxSpMovememF3dex2(index, offset,
                                Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash));
    } else {
        auto light = (Fast::LightEntry*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash);
        uintptr_t data = (uintptr_t)&light->Ambient;
        gfx->GfxSpMovememF3d(index, offset, (void*)(data + (hasOffset == 1 ? 0x8 : 0)));
    }
    return false;
}

bool gfx_push_shader(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    const char* path = (const char*)gfx->SegAddr(cmd->words.w1);

    if (!gfx_check_image_signature(path)) {
        SPDLOG_ERROR("G_PUSH_SHADER: Shader is not a valid OTR resource name, unable to register push shader");
        return false;
    }

    path = &path[7];

    size_t shaderId = static_cast<size_t>(-1);
    for (const auto& shader : gfx->mShaders) {
        if (strcmp(shader.second, path) == 0) {
            shaderId = shader.first;
            break;
        }
    }

    if (shaderId == static_cast<size_t>(-1)) {
        shaderId = gfx->mShadersIndex++;
        gfx->mShaders[shaderId] = path;
    }

    gfx->mShaderStack.push(shaderId);

    return false;
}

bool gfx_pop_shader(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mShaderStack.pop();

    return false;
}

const char* gfx_get_shader(int16_t id) {
    Interpreter* gfx = mInstance.lock().get();

    for (const std::pair<size_t, const char*>& shader : gfx->mShaders) {
        if (shader.first == id) {
            return shader.second;
        }
    }

    return nullptr; // Use no shader
}

bool gfx_moveword_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovewordF3dex2(C0(16, 8), C0(0, 16), cmd->words.w1);

    return false;
}

bool gfx_moveword_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpMovewordF3d(C0(0, 8), C0(8, 16), cmd->words.w1);

    return false;
}

bool gfx_texture_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTexture(C1(16, 16), C1(0, 16), C0(11, 3), C0(8, 3), C0(1, 7));

    return false;
}

// Seems to be the same for all other non F3DEX2 microcodes...
bool gfx_texture_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTexture(C1(16, 16), C1(0, 16), C0(11, 3), C0(8, 3), C0(0, 8));

    return false;
}

// Almost all versions of the microcode have their own version of this opcode
bool gfx_vtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpVertex(C0(12, 8), C0(1, 7) - C0(12, 8), (const F3DVtx*)gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_vtx_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    gfx->GfxSpVertex(C0(10, 6), C0(17, 7), (const F3DVtx*)gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_vtx_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpVertex((C0(0, 16)) / sizeof(F3DVtx), C0(16, 4), (const F3DVtx*)gfx->SegAddr(cmd->words.w1));

    return false;
}

bool gfx_vtx_hash_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    // Offset added to the start of the vertices
    const uintptr_t offset = (*cmd0)->words.w1;
    // This is a two-part display list command, so increment the instruction pointer so we can get the CRC64
    // hash from the second
    (*cmd0)++;
    const uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

    // We need to know if the offset is a cached pointer or not. An offset greater than one million is not a
    // real offset, so it must be a real pointer
    if (offset > 0xFFFFF) {
        (*cmd0)--;
        F3DGfx* cmd = *cmd0;
        gfx->GfxSpVertex(C0(12, 8), C0(1, 7) - C0(12, 8), (F3DVtx*)offset);
        (*cmd0)++;
    } else {
        F3DVtx* vtx = (F3DVtx*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash);

        if (vtx != NULL) {
            vtx = (F3DVtx*)((char*)vtx + offset);

            (*cmd0)--;
            F3DGfx* cmd = *cmd0;

            // TODO: WTF??
            cmd->words.w1 = (uintptr_t)vtx;

            gfx->GfxSpVertex(C0(12, 8), C0(1, 7) - C0(12, 8), vtx);
            (*cmd0)++;
        }
    }
    return false;
}

bool gfx_vtx_otr_filepath_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    char* fileName = (char*)cmd->words.w1;
    (*cmd0)++;
    cmd = *cmd0;
    size_t vtxCnt = cmd->words.w0;
    size_t vtxIdxOff = cmd->words.w1 >> 16;
    size_t vtxDataOff = cmd->words.w1 & 0xFFFF;
    F3DVtx* vtx =
        (F3DVtx*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer((const char*)fileName);
    vtx += vtxDataOff;

    gfx->GfxSpVertex(vtxCnt, vtxIdxOff, vtx);
    return false;
}

bool gfx_dl_otr_filepath_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    char* fileName = (char*)cmd->words.w1;
    F3DGfx* nDL =
        (F3DGfx*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer((const char*)fileName);

    if (C0(16, 1) == 0 && nDL != nullptr) {
        g_exec_stack.call(*cmd0, nDL);
    } else {
        if (nDL != nullptr) {
            (*cmd0) = nDL;
            g_exec_stack.branch(cmd);
            return true; // shortcut cmd increment
        } else {
            assert(0 && "???");
            // gfx_path.pop_back();
            // cmd = cmd_stack.top();
            // cmd_stack.pop();
        }
    }
    return false;
}

// The original F3D microcode doesn't seem to have this opcode. Glide handles it as part of moveword
bool gfx_modify_vtx_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    gfx->GfxSpModifyVertex(C0(1, 15), C0(16, 8), (uint32_t)cmd->words.w1);
    return false;
}

// F3D, F3DEX, and F3DEX2 do the same thing but F3DEX2 has its own opcode number
bool gfx_dl_handler_common(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    F3DGfx* subGFX = (F3DGfx*)gfx->SegAddr(cmd->words.w1);
    if (C0(16, 1) == 0) {
        // Push return address
        if (subGFX != nullptr) {
            g_exec_stack.call(*cmd0, subGFX);
        }
    } else {
        (*cmd0) = subGFX;
        g_exec_stack.branch(cmd);
        return true; // shortcut cmd increment
    }
    return false;
}

bool gfx_dl_otr_hash_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    if (C0(16, 1) == 0) {
        // Push return address
        (*cmd0)++;

        uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

        F3DGfx* gfx = (F3DGfx*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash);

        if (gfx != 0) {
            g_exec_stack.call(cmd, gfx);
        }
    } else {
        Interpreter* gfx = mInstance.lock().get();
        assert(0 && "????");
        (*cmd0) = (F3DGfx*)gfx->SegAddr((*cmd0)->words.w1);
        return true;
    }
    return false;
}
bool gfx_dl_index_handler(F3DGfx** cmd0) {
    // Compute seg addr by converting an index value to a offset value
    // handling 32 vs 64 bit size differences for Gfx
    // adding 1 to trigger the segaddr flow
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = (*cmd0);
    uint8_t segNum = (uint8_t)(cmd->words.w1 >> 24);
    uint32_t index = (uint32_t)(cmd->words.w1 & 0x00FFFFFF);
    uintptr_t segAddr = (segNum << 24) | (index * sizeof(F3DGfx)) + 1;

    F3DGfx* subGFX = (F3DGfx*)gfx->SegAddr(segAddr);
    if (C0(16, 1) == 0) {
        // Push return address
        if (subGFX != nullptr) {
            g_exec_stack.call((*cmd0), subGFX);
        }
    } else {
        (*cmd0) = subGFX;
        g_exec_stack.branch(cmd);
        return true; // shortcut cmd increment
    }
    return false;
}

// TODO handle special OTR opcodes later...
bool gfx_pushcd_handler_custom(F3DGfx** cmd0) {
    gfx_push_current_dir((char*)(*cmd0)->words.w1);
    return false;
}

// TODO handle special OTR opcodes later...
bool gfx_branch_z_otr_handler_f3dex2(F3DGfx** cmd0) {
    // Push return address
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = (*cmd0);

    uint8_t vbidx = (uint8_t)((*cmd0)->words.w0 & 0x00000FFF);
    uint32_t zval = (uint32_t)((*cmd0)->words.w1);

    (*cmd0)++;

    if (gfx->mRsp->loaded_vertices[vbidx].z <= zval ||
        (gfx->mRsp->extra_geometry_mode & G_EX_ALWAYS_EXECUTE_BRANCH) != 0) {
        uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (*cmd0)->words.w1;

        F3DGfx* gfx = (F3DGfx*)Ship::Context::GetInstance()->GetResourceManager()->GetResourceRawPointer(hash);

        if (gfx != 0) {
            (*cmd0) = gfx;
            g_exec_stack.branch(cmd);
            return true; // shortcut cmd increment
        }
    }
    return false;
}

bool gfx_rdphalf_1_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    gfx->mRsp->branch_z_target = (*cmd0)->words.w1;
    return false;
}

bool gfx_branch_z_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* caller = *cmd0;
    const uint8_t vbidx = static_cast<uint8_t>((caller->words.w0 & 0xFFFu) >> 1);
    const uint32_t zval = static_cast<uint32_t>(caller->words.w1);
    F3DGfx* target = reinterpret_cast<F3DGfx*>(gfx->mRsp->branch_z_target);
    gfx->mRsp->branch_z_target = 0;
    if (target == nullptr || vbidx >= MAX_VERTICES) {
        return false;
    }
    const LoadedVertex& vertex = gfx->mRsp->loaded_vertices[vbidx];
    const float screenZ =
        (fabsf(vertex.w) > 0.000001f)
            ? ((vertex.z / vertex.w) * gfx->mRsp->viewport_z_scale + gfx->mRsp->viewport_z_trans)
            : gfx->mRsp->viewport_z_trans;
    const uint32_t screenZFixed =
        static_cast<uint32_t>(std::clamp(screenZ, 0.0f, 65535.0f) * 65536.0f);

    if (screenZFixed <= zval ||
        (gfx->mRsp->extra_geometry_mode & G_EX_ALWAYS_EXECUTE_BRANCH) != 0) {
        *cmd0 = target;
        g_exec_stack.branch(caller);
        return true;
    }
    return false;
}

// F3D, F3DEX, and F3DEX2 do the same thing but F3DEX2 has its own opcode number
bool gfx_end_dl_handler_common(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    gfx->mMarkerOn = false;
    g_exec_stack.ret();
    return true;
}

bool gfx_set_prim_depth_handler_rdp(F3DGfx** cmd) {
    Interpreter* gfx = mInstance.lock().get();
    uint32_t w1 = (*cmd)->words.w1;
    const uint16_t newPrimDepth = (uint16_t)((w1 >> 16) & 0x7FFF); // Mask to 15 bits
    // [prim-depth-flush] Flush() samples mRdp->prim_depth lazily at drain time, so buffered
    // geometry must drain before a new value overwrites the register, or it flushes with state it
    // was never built with (the racer.c rival-icon bug). Detecting the change in GfxSpTri1 is too
    // late; the register is already overwritten. GDX_NO_PRIM_DEPTH_FLUSH=1 suppresses the drain.
    static const bool sNoPrimDepthFlush = [] {
        const char* e = std::getenv("GDX_NO_PRIM_DEPTH_FLUSH");
        return e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    if (!sNoPrimDepthFlush && newPrimDepth != gfx->mRdp->prim_depth) {
        gfx->Flush();
    }
    gfx->mRdp->prim_depth = newPrimDepth;
    return false;
}

// Only on F3DEX2
bool gfx_geometry_mode_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode(~C0(0, 24), (uint32_t)cmd->words.w1);
    return false;
}

// Only on F3DEX and older
bool gfx_set_geometry_mode_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode(0, (uint32_t)cmd->words.w1);
    return false;
}

// Only on F3DEX and older
bool gfx_clear_geometry_mode_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode((uint32_t)cmd->words.w1, 0);
    return false;
}

bool gfx_tri1_otr_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();

    F3DGfx* cmd = *cmd0;
    uint8_t v00 = (uint8_t)(cmd->words.w0 & 0x0000FFFF);
    uint8_t v01 = (uint8_t)(cmd->words.w1 >> 16);
    uint8_t v02 = (uint8_t)(cmd->words.w1 & 0x0000FFFF);
    gfx->GfxSpTri1(v00, v01, v02, false);

    return false;
}

bool gfx_tri1_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C0(16, 8) / 2, C0(8, 8) / 2, C0(0, 8) / 2, false);

    return false;
}

bool gfx_tri1_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C1(17, 7), C1(9, 7), C1(1, 7), false);

    return false;
}

bool gfx_tri1_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C1(16, 8) / 10, C1(8, 8) / 10, C1(0, 8) / 10, false);

    return false;
}

// F3DEX, and F3DEX2 share a tri2 function, however F3DEX has a different quad function.
bool gfx_tri2_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C0(17, 7), C0(9, 7), C0(1, 7), false);
    gfx->GfxSpTri1(C1(17, 7), C1(9, 7), C1(1, 7), false);
    return false;
}

bool gfx_quad_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C0(16, 8) / 2, C0(8, 8) / 2, C0(0, 8) / 2, false);
    gfx->GfxSpTri1(C1(16, 8) / 2, C1(8, 8) / 2, C1(0, 8) / 2, false);
    return false;
}

bool gfx_quad_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpTri1(C1(16, 8) / 2, C1(8, 8) / 2, C1(0, 8) / 2, false);
    gfx->GfxSpTri1(C1(16, 8) / 2, C1(0, 8) / 2, C1(24, 8) / 2, false);
    return false;
}

bool gfx_othermode_l_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(31 - C0(8, 8) - C0(0, 8), C0(0, 8) + 1, cmd->words.w1);

    return false;
}

bool gfx_othermode_l_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(C0(8, 8), C0(0, 8), cmd->words.w1);

    return false;
}

bool gfx_othermode_h_handler_f3dex2(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(63 - C0(8, 8) - C0(0, 8), C0(0, 8) + 1, (uint64_t)cmd->words.w1 << 32);

    return false;
}

// Only on F3DEX and older
bool gfx_set_geometry_mode_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode(0, (uint32_t)cmd->words.w1);
    return false;
}

// Only on F3DEX and older
bool gfx_clear_geometry_mode_handler_f3dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpGeometryMode((uint32_t)cmd->words.w1, 0);
    return false;
}

bool gfx_othermode_h_handler_f3d(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxSpSetOtherMode(C0(8, 8) + 32, C0(0, 8), (uint64_t)cmd->words.w1 << 32);

    return false;
}

bool gfx_set_timg_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    uintptr_t i = (uintptr_t)gfx->SegAddr(cmd->words.w1);

    char* imgData = (char*)i;
    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetdata = {};
    // Default scale factors to 1 for raw N64 textures. OTR textures set these
    // from the resource, but raw textures would leave them at 0.
    rawTexMetdata.h_byte_scale = 1;
    rawTexMetdata.v_pixel_scale = 1;

    if ((i & 1) != 1) {
        if (gfx_check_image_signature(imgData) == 1) {
            std::shared_ptr<Fast::Texture> tex = std::static_pointer_cast<Fast::Texture>(
                Ship::Context::GetInstance()->GetResourceManager()->LoadResourceProcess(imgData));

            if (tex == nullptr) {
                (*cmd0)++;
                return false;
            }

            i = (uintptr_t) reinterpret_cast<char*>(tex->ImageData);
            texFlags = tex->Flags;
            rawTexMetdata.width = tex->Width;
            rawTexMetdata.height = tex->Height;
            rawTexMetdata.h_byte_scale = tex->HByteScale;
            rawTexMetdata.v_pixel_scale = tex->VPixelScale;
            rawTexMetdata.type = tex->Type;
            rawTexMetdata.resource = tex;
        }
    }

    // If the resolved address is still in the N64 segmented range, SegAddr
    // failed to resolve it (segment not set up). Skip to avoid dereferencing
    // invalid memory.
    // For Windows, also check if the address is not from a dll because this validation returns a false positive caused
    // by how the virtual memory is allocated.
#ifdef _WIN32
    HMODULE module = nullptr;
    if (i <= 0x0FFFFFFF &&
        !(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                             reinterpret_cast<LPCSTR>(i), &module))) {
        return false;
    }
#else
    if (i <= 0x0FFFFFFF) {
        return false;
    }
#endif

    gfx->GfxDpSetTextureImage(C0(21, 3), C0(19, 2), C0(0, 12) + 1, imgData, texFlags, rawTexMetdata, (void*)i);

    return false;
}

bool gfx_set_timg_otr_hash_handler_custom(F3DGfx** cmd0) {
    uintptr_t addr = (*cmd0)->words.w1;
    (*cmd0)++;
    uint64_t hash = ((uint64_t)(*cmd0)->words.w0 << 32) + (uint64_t)(*cmd0)->words.w1;

    const char* fileName = Ship::Context::GetInstance()->GetResourceManager()->GetArchiveManager()->HashToCString(hash);
    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    if (fileName == nullptr) {
        (*cmd0)++;
        return false;
    }

    std::shared_ptr<Fast::Texture> texture =
        std::static_pointer_cast<Fast::Texture>(Ship::Context::GetInstance()->GetResourceManager()->LoadResourceProcess(
            Ship::Context::GetInstance()->GetResourceManager()->GetArchiveManager()->HashToCString(hash)));
    if (texture != nullptr) {
        texFlags = texture->Flags;
        rawTexMetadata.width = texture->Width;
        rawTexMetadata.height = texture->Height;
        rawTexMetadata.h_byte_scale = texture->HByteScale;
        rawTexMetadata.v_pixel_scale = texture->VPixelScale;
        rawTexMetadata.type = texture->Type;
        rawTexMetadata.resource = texture;

        // OTRTODO: We have disabled caching for now to fix a texture corruption issue with HD texture
        // support. In doing so, there is a potential performance hit since we are not caching lookups. We
        // need to do proper profiling to see whether or not it is worth it to keep the caching system.

        char* tex = reinterpret_cast<char*>(texture->ImageData);

        if (tex != nullptr) {
            (*cmd0)--;
            uintptr_t oldData = (*cmd0)->words.w1;
            // TODO: wtf??
            (*cmd0)->words.w1 = (uintptr_t)tex;

            // if (ourHash != (uint64_t)-1) {
            //     auto res = ResourceLoad(ourHash);
            // }

            (*cmd0)++;
        }

        (*cmd0)--;
        F3DGfx* cmd = (*cmd0);
        uint32_t fmt = C0(21, 3);
        uint32_t size = C0(19, 2);
        uint32_t width = C0(0, 12) + 1;

        if (tex != NULL) {
            Interpreter* gfx = mInstance.lock().get();
            gfx->GfxDpSetTextureImage(fmt, size, width, fileName, texFlags, rawTexMetadata, tex);
        }
    } else {
        SPDLOG_ERROR("G_SETTIMG_OTR_HASH: Texture is null");
    }

    (*cmd0)++;
    return false;
}

bool gfx_set_timg_otr_filepath_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    const char* fileName = (char*)cmd->words.w1;

    uint32_t texFlags = 0;
    RawTexMetadata rawTexMetadata = {};

    std::shared_ptr<Fast::Texture> texture = std::static_pointer_cast<Fast::Texture>(
        Ship::Context::GetInstance()->GetResourceManager()->LoadResourceProcess(fileName));
    if (texture != nullptr) {
        Interpreter* gfx = mInstance.lock().get();
        texFlags = texture->Flags;
        rawTexMetadata.width = texture->Width;
        rawTexMetadata.height = texture->Height;
        rawTexMetadata.h_byte_scale = texture->HByteScale;
        rawTexMetadata.v_pixel_scale = texture->VPixelScale;
        rawTexMetadata.type = texture->Type;
        rawTexMetadata.resource = texture;

        uint32_t fmt = C0(21, 3);
        uint32_t size = C0(19, 2);
        uint32_t width = C0(0, 12) + 1;

        gfx->GfxDpSetTextureImage(fmt, size, width, fileName, texFlags, rawTexMetadata,
                                  reinterpret_cast<char*>(texture->ImageData));
    } else {
        static uint32_t sMissingTextureLogs = 0;
        if (sMissingTextureLogs < 8) {
            ++sMissingTextureLogs;
            SPDLOG_ERROR("G_SETTIMG_OTR_FILEPATH: Texture '{}' is null", fileName);
            if (sMissingTextureLogs == 8) {
                SPDLOG_ERROR("G_SETTIMG_OTR_FILEPATH: further missing-texture errors suppressed");
            }
        }
    }
    return false;
}

bool gfx_set_fb_handler_custom(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();
    gfx->Flush();

    if (cmd->words.w1) {
        gfx->SetFrameBuffer((int32_t)cmd->words.w1, 1.0f);
        gfx->mActiveFrameBuffer = gfx->mFrameBuffers.find((int32_t)cmd->words.w1);
        gfx->mFbActive = true;
    } else {
        gfx->ResetFrameBuffer();
        gfx->mFbActive = false;
        gfx->mActiveFrameBuffer = gfx->mFrameBuffers.end();
    }
    return false;
}

bool gfx_reset_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    gfx->Flush();
    gfx->mFbActive = false;
    gfx->mActiveFrameBuffer = gfx->mFrameBuffers.end();
    gfx->mRapi->StartDrawToFramebuffer(gfx->mRendersToFb ? gfx->mGameFb : 0,
                                       (float)gfx->mCurDimensions.height / gfx->mNativeDimensions.height);
    // Force viewport and scissor to reapply against the main framebuffer, in case a previous smaller
    // framebuffer truncated the values
    gfx->mRdp->viewport_or_scissor_changed = true;
    gfx->mRenderingState.viewport = {};
    gfx->mRenderingState.scissor = {};
    return false;
}

bool gfx_copy_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    bool* hasCopiedPtr = (bool*)cmd->words.w1;

    gfx->Flush();
    gfx->CopyFrameBuffer(C0(11, 11), C0(0, 11), (bool)C0(22, 1), hasCopiedPtr);
    return false;
}

bool gfx_read_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    int32_t width, height;
    [[maybe_unused]] int32_t ulx, uly;
    uint16_t* rgba16Buffer = (uint16_t*)cmd->words.w1;
    int fbId = C0(0, 8);
    bool bswap = C0(8, 1);
    ++(*cmd0);
    cmd = *cmd0;
    // Specifying the upper left origin value is unused and unsupported at the renderer level
    ulx = C0(0, 16);
    uly = C0(16, 16);
    width = C1(0, 16);
    height = C1(16, 16);

    gfx->Flush();
    gfx->mRapi->ReadFramebufferToCPU(fbId, width, height, rgba16Buffer);

#ifndef IS_BIGENDIAN
    // byteswap the output to BE
    if (bswap) {
        for (size_t i = 0; i < (size_t)width * height; i++) {
            rgba16Buffer[i] = BE16SWAP(rgba16Buffer[i]);
        }
    }
#endif

    return false;
}

bool gfx_register_blended_texture_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    // Flush incase we are replacing a previous blended texture that hasn't been finialized to the GPU
    gfx->Flush();

    char* timg = (char*)cmd->words.w1;

    ++(*cmd0);
    cmd = *cmd0;

    uint8_t* mask = (uint8_t*)cmd->words.w0;
    uint8_t* replacementTex = (uint8_t*)cmd->words.w1;

    if (!gfx_check_image_signature(timg)) {
        SPDLOG_ERROR(
            "OTR_G_REGBLENDEDTEX: Texture is not a valid OTR resource name, unable to register blended texture");
        return false;
    }

    // With no mask, we should clear the blended texture
    if (mask == nullptr) {
        gfx->UnregisterBlendedTexture(timg);
    } else {
        gfx->RegisterBlendedTexture(timg, mask, replacementTex);
    }

    return false;
}

bool gfx_set_timg_fb_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->Flush();
    gfx->mRapi->SelectTextureFb((uint32_t)cmd->words.w1);
    gfx->mRdp->textures_changed[0] = false;
    gfx->mRdp->textures_changed[1] = false;
    return false;
}

bool gfx_set_grayscale_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mRdp->grayscale = cmd->words.w1;
    return false;
}

bool gfx_load_block_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpLoadBlock(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_load_block_wide_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    uint32_t tile = cmd->words.w0 & 0x7;
    uint32_t lrs = cmd->words.w1;

    (*cmd0)++;
    cmd = *cmd0;

    uint32_t uls = (cmd->words.w0 >> 16) & 0xFFFF;
    uint32_t ult = (cmd->words.w0 >> 0) & 0xFFFF;
    uint32_t dxt = (cmd->words.w1 >> 0) & 0xFFF;

    gfx->GfxDpLoadBlock(tile, uls, ult, lrs, dxt);
    return false;
}

bool gfx_load_tile_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpLoadTile(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_set_tile_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetTile(C0(21, 3), C0(19, 2), C0(9, 9), C0(0, 9), C1(24, 3), C1(20, 4), C1(18, 2), C1(14, 4), C1(10, 4),
                      C1(8, 2), C1(4, 4), C1(0, 4));
    return false;
}

bool gfx_set_tile_size_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetTileSize(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_set_tile_size_interp_handler_rdp(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();

    if (gfx->mInterpolationIndex == gfx->mInterpolationIndexTarget) {
        int tile = C1(24, 3);
        gfx->GfxDpSetTileSize(C1(24, 3), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
        ++(*cmd0);
        memcpy(&gfx->mRdp->texture_tile[tile].uls, &(*cmd0)->words.w0, sizeof(float));
        memcpy(&gfx->mRdp->texture_tile[tile].ult, &(*cmd0)->words.w1, sizeof(float));
        ++(*cmd0);
        memcpy(&gfx->mRdp->texture_tile[tile].lrs, &(*cmd0)->words.w0, sizeof(float));
        memcpy(&gfx->mRdp->texture_tile[tile].lrt, &(*cmd0)->words.w1, sizeof(float));
    } else {
        ++(*cmd0);
        ++(*cmd0);
    }

    return false;
}

bool gfx_set_interpolation_index_target(F3DGfx** cmd0) {
    F3DGfx* cmd = *cmd0;
    Interpreter* gfx = mInstance.lock().get();

    gfx->mInterpolationIndexTarget = cmd->words.w1;
    return false;
}

bool gfx_load_tlut_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpLoadTlut(C1(24, 3), C1(14, 10));
    return false;
}

bool gfx_set_env_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetEnvColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_prim_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetPrimColor(C0(8, 8), C0(0, 8), C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_fog_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetFogColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

// CENTER/SCALE and K4/K5 are wired as combiner inputs, so the standard (A-B)*C+D
// shader path covers their common uses. TODO: chroma-key width/threshold
// gating from G_SETKEYR/GB (wR/wG/wB ignored) and the YUV->RGB matrix K0..K3
// applied during texture sampling.
// G_SETKEYR: w1 = [wR:12 | cR:8 | sR:8]
bool gfx_set_key_r_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mRdp->key_center.r = C1(8, 8);
    gfx->mRdp->key_scale.r = C1(0, 8);
    return false;
}

// G_SETKEYGB: w0 = [op:8 | wG:12 | _:4 | wB:12], w1 = [cG:8 | sG:8 | cB:8 | sB:8]
bool gfx_set_key_gb_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mRdp->key_center.g = C1(24, 8);
    gfx->mRdp->key_scale.g = C1(16, 8);
    gfx->mRdp->key_center.b = C1(8, 8);
    gfx->mRdp->key_scale.b = C1(0, 8);
    return false;
}

// G_SETCONVERT: w0 = [op:8 | k0:9 | k1:9 | k2_hi:4], w1 = [k2_lo:5 | k3:9 | k4:9 | k5:9]
// K0..K5 are signed 9-bit values; sign-extend after decoding.
bool gfx_set_convert_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->mRdp->convert_k[0] = sign_extend_9(C0(13, 9));
    gfx->mRdp->convert_k[1] = sign_extend_9(C0(4, 9));
    // k2 is split across w0 and w1
    gfx->mRdp->convert_k[2] = sign_extend_9((C0(0, 4) << 5) | C1(27, 5));
    gfx->mRdp->convert_k[3] = sign_extend_9(C1(18, 9));
    gfx->mRdp->convert_k[4] = sign_extend_9(C1(9, 9));
    gfx->mRdp->convert_k[5] = sign_extend_9(C1(0, 9));
    return false;
}

bool gfx_set_blend_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    // [blend-alpha-flush] Same hazard as [prim-depth-flush]: Flush() samples blend_color.a lazily
    // for the alpha-compare threshold, so buffered geometry must drain before the register
    // changes. Only alpha matters; r/g/b never reach Flush(). GDX_NO_BLEND_ALPHA_FLUSH=1
    // suppresses the drain.
    const uint8_t newBlendAlpha = (uint8_t)C1(0, 8);
    static const bool sNoBlendAlphaFlush = [] {
        const char* e = std::getenv("GDX_NO_BLEND_ALPHA_FLUSH");
        return e != nullptr && e[0] != '\0' && strcmp(e, "0") != 0;
    }();
    if (!sNoBlendAlphaFlush && newBlendAlpha != gfx->mRdp->blend_color.a) {
        gfx->Flush();
    }
    gfx->GfxDpSetBlendColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_fill_color_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetFillColor((uint32_t)cmd->words.w1);
    return false;
}

bool gfx_set_intensity_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetGrayscaleColor(C1(24, 8), C1(16, 8), C1(8, 8), C1(0, 8));
    return false;
}

bool gfx_set_combine_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;

    gfx->GfxDpSetCombineMode(
        color_comb(C0(20, 4), C1(28, 4), C0(15, 5), C1(15, 3)), alpha_comb(C0(12, 3), C1(12, 3), C0(9, 3), C1(9, 3)),
        color_comb(C0(5, 4), C1(24, 4), C0(0, 5), C1(6, 3)), alpha_comb(C1(21, 3), C1(3, 3), C1(18, 3), C1(0, 3)));
    return false;
}

bool gfx_tex_rect_and_flip_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    int8_t opcode = (int8_t)(cmd->words.w0 >> 24);
    int32_t lrx, lry, tile, ulx, uly;
    uint32_t uls, ult, dsdx, dtdy;

    lrx = C0(12, 12);
    lry = C0(0, 12);
    tile = C1(24, 3);
    ulx = C1(12, 12);
    uly = C1(0, 12);
    // TODO make sure I don't need to increment cmd0
    ++(*cmd0);
    cmd = *cmd0;
    uls = C1(16, 16);
    ult = C1(0, 16);
    ++(*cmd0);
    cmd = *cmd0;
    dsdx = C1(16, 16);
    dtdy = C1(0, 16);

    gfx->GfxDpTextureRectangle(ulx, uly, lrx, lry, tile, uls, ult, dsdx, dtdy, opcode == RDP_G_TEXRECTFLIP);
    return false;
}

bool gfx_tex_rect_wide_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    int8_t opcode = (int8_t)(cmd->words.w0 >> 24);
    int32_t lrx, lry, tile, ulx, uly;
    uint32_t uls, ult, dsdx, dtdy;

    lrx = static_cast<int32_t>((C0(0, 24) << 8)) >> 8;
    lry = static_cast<int32_t>((C1(0, 24) << 8)) >> 8;
    tile = C1(24, 3);
    ++(*cmd0);
    cmd = *cmd0;
    ulx = static_cast<int32_t>((C0(0, 24) << 8)) >> 8;
    uly = static_cast<int32_t>((C1(0, 24) << 8)) >> 8;
    ++(*cmd0);
    cmd = *cmd0;
    uls = C0(16, 16);
    ult = C0(0, 16);
    dsdx = C1(16, 16);
    dtdy = C1(0, 16);
    gfx->GfxDpTextureRectangle(ulx, uly, lrx, lry, tile, uls, ult, dsdx, dtdy, opcode == RDP_G_TEXRECTFLIP);
    return false;
}

bool gfx_image_rect_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *cmd0;
    int16_t tile, iw, ih;
    int16_t x0, y0, s0, t0;
    int16_t x1, y1, s1, t1;
    tile = C0(0, 3);
    iw = C1(16, 16);
    ih = C1(0, 16);
    cmd = ++(*cmd0);
    x0 = C0(16, 16);
    y0 = C0(0, 16);
    s0 = C1(16, 16);
    t0 = C1(0, 16);
    cmd = ++(*cmd0);
    x1 = C0(16, 16);
    y1 = C0(0, 16);
    s1 = C1(16, 16);
    t1 = C1(0, 16);
    gfx->GfxDpImageRectangle(tile, iw, ih, x0, y0, s0, t0, x1, y1, s1, t1);

    return false;
}

bool gfx_fill_rect_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpFillRectangle(C1(12, 12), C1(0, 12), C0(12, 12), C0(0, 12));
    return false;
}

bool gfx_fill_wide_rect_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);
    int32_t lrx, lry, ulx, uly;

    lrx = (int32_t)(C0(0, 24) << 8) >> 8;
    lry = (int32_t)(C1(0, 24) << 8) >> 8;
    cmd = ++(*cmd0);
    ulx = (int32_t)(C0(0, 24) << 8) >> 8;
    uly = (int32_t)(C1(0, 24) << 8) >> 8;
    gfx->GfxDpFillRectangle(ulx, uly, lrx, lry);

    return false;
}

bool gfx_SetScissor_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetScissor(C1(24, 2), C0(12, 12), C0(0, 12), C1(12, 12), C1(0, 12));
    return false;
}

bool gfx_set_z_img_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetZImage(gfx->SegAddr(cmd->words.w1));
    return false;
}

bool gfx_set_c_img_handler_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetColorImage(C0(21, 3), C0(19, 2), C0(0, 11), gfx->SegAddr(cmd->words.w1));
    return false;
}

bool gfx_rdp_set_other_mode_rdp(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxDpSetOtherMode(C0(0, 24), (uint32_t)cmd->words.w1);
    return false;
}

bool gfx_bg_copy_handler_s2dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    if (!gfx->mMarkerOn) {
        gfx->Gfxs2dexBgCopy((F3DuObjBg*)cmd->words.w1); // not gfx->SegAddr here it seems
    }
    return false;
}

bool gfx_bg_1cyc_handler_s2dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->Gfxs2dexBg1cyc((F3DuObjBg*)cmd->words.w1);
    return false;
}

bool gfx_obj_rectangle_handler_s2dex(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    if (!gfx->mMarkerOn) {
        gfx->Gfxs2dexRecyCopy((F3DuObjSprite*)cmd->words.w1); // not gfx->SegAddr here it seems
    }
    return false;
}

bool gfx_extra_geometry_mode_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    gfx->GfxSpExtraGeometryMode(~C0(0, 24), (uint32_t)cmd->words.w1);
    return false;
}

// L3DEX2 G_LINE3D rewritten by the port's gfx bridge. Operand encoding is preserved: vertex
// indices arrive pre-multiplied by 2, exactly like native L3DEX2.
bool gfx_line3d_gdx_handler_custom(F3DGfx** cmd0) {
    Interpreter* gfx = mInstance.lock().get();
    F3DGfx* cmd = *(cmd0);

    const uint8_t v0 = (uint8_t)(C0(16, 8) / 2);
    const uint8_t v1 = (uint8_t)(C0(8, 8) / 2);
    const uint8_t wd = (uint8_t)C0(0, 8);
    gfx->GfxSpLine3DGdx(v0, v1, wd);
    return false;
}

bool gfx_stubbed_command_handler(F3DGfx** cmd0) {
    return false;
}

// gDPPipeSync: hardware drains its pipeline here so later state changes cannot retroactively
// affect queued draws. The drain was once removed as redundant with the source-side
// [prim-depth-flush]/[blend-alpha-flush] guards, but removing it coincided with the Mute City sky
// alternating every other tick. GDX_NO_PIPESYNC_FLUSH=1 restores the no-drain behaviour.
bool gfx_rdp_pipe_sync_handler_rdp(F3DGfx** cmd0) {
    static const bool sNoFlush = [] {
        const char* e = std::getenv("GDX_NO_PIPESYNC_FLUSH");
        return e != nullptr && e[0] != 0 && !(e[0] == '0' && e[1] == 0);
    }();
    if (!sNoFlush) {
        Interpreter* gfx = mInstance.lock().get();
        gfx->Flush();
    }
    return false;
}

bool gfx_spnoop_command_handler_f3dex2(F3DGfx** cmd0) {
    return false;
}

class UcodeHandler {
  public:
    inline constexpr UcodeHandler(
        std::initializer_list<std::pair<int8_t, std::pair<const char*, GfxOpcodeHandlerFunc>>> initializer) {
        std::fill(std::begin(mHandlers), std::end(mHandlers),
                  std::pair<const char*, GfxOpcodeHandlerFunc>(nullptr, nullptr));

        for (const auto& [opcode, handler] : initializer) {
            mHandlers[static_cast<uint8_t>(opcode)] = handler;
        }
    }

    inline bool contains(int8_t opcode) const {
        return mHandlers[static_cast<uint8_t>(opcode)].first != nullptr;
    }

    inline std::pair<const char*, GfxOpcodeHandlerFunc> at(int8_t opcode) const {
        return mHandlers[static_cast<uint8_t>(opcode)];
    }

  private:
    std::pair<const char*, GfxOpcodeHandlerFunc> mHandlers[std::numeric_limits<uint8_t>::max() + 1];
};

static constexpr UcodeHandler rdpHandlers = {
    { RDP_G_SETTARGETINTERPINDEX,
      { "G_SETTARGETINTERPINDEX", gfx_set_interpolation_index_target } }, // G_SETTARGETINTERPINDEX
    { RDP_G_SETTILESIZE_INTERP,
      { "G_SETTILESIZE_INTERP", gfx_set_tile_size_interp_handler_rdp } },            // G_SETTILESIZE_INTERP
    { RDP_G_TEXRECT, { "G_TEXRECT", gfx_tex_rect_and_flip_handler_rdp } },           // G_TEXRECT (-28)
    { RDP_G_TEXRECTFLIP, { "G_TEXRECTFLIP", gfx_tex_rect_and_flip_handler_rdp } },   // G_TEXRECTFLIP (-27)
    { RDP_G_RDPLOADSYNC, { "mRdpLOADSYNC", gfx_stubbed_command_handler } },          // mRdpLOADSYNC (-26)
    // Real Flush(), not a stub: the no-op let state changes made after a sync
    // retroactively affect draws queued before it (see gfx_rdp_pipe_sync_handler_rdp).
    { RDP_G_RDPPIPESYNC, { "mRdpPIPESYNC", gfx_rdp_pipe_sync_handler_rdp } },        // mRdpPIPESYNC (-25)
    { RDP_G_RDPTILESYNC, { "mRdpTILESYNC", gfx_stubbed_command_handler } },          // mRdpPIPESYNC (-24)
    { RDP_G_RDPFULLSYNC, { "mRdpFULLSYNC", gfx_stubbed_command_handler } },          // mRdpFULLSYNC (-23)
    { RDP_G_SETKEYGB, { "G_SETKEYGB", gfx_set_key_gb_handler_rdp } },                // G_SETKEYGB (-22)
    { RDP_G_SETKEYR, { "G_SETKEYR", gfx_set_key_r_handler_rdp } },                   // G_SETKEYR (-21)
    { RDP_G_SETCONVERT, { "G_SETCONVERT", gfx_set_convert_handler_rdp } },           // G_SETCONVERT (-20)
    { RDP_G_SETSCISSOR, { "G_SETSCISSOR", gfx_SetScissor_handler_rdp } },            // G_SETSCISSOR (-19)
    { RDP_G_SETPRIMDEPTH, { "G_SETPRIMDEPTH", gfx_set_prim_depth_handler_rdp } },    // G_SETPRIMDEPTH (-18)
    { RDP_G_RDPSETOTHERMODE, { "mRdpSETOTHERMODE", gfx_rdp_set_other_mode_rdp } },   // mRdpSETOTHERMODE (-17)
    { RDP_G_LOADTLUT, { "G_LOADTLUT", gfx_load_tlut_handler_rdp } },                 // G_LOADTLUT (-16)
    { RDP_G_SETTILESIZE, { "G_SETTILESIZE", gfx_set_tile_size_handler_rdp } },       // G_SETTILESIZE (-14)
    { RDP_G_LOADBLOCK, { "G_LOADBLOCK", gfx_load_block_handler_rdp } },              // G_LOADBLOCK (-13)
    { RDP_G_LOADTILE, { "G_LOADTILE", gfx_load_tile_handler_rdp } },                 // G_LOADTILE (-12)
    { RDP_G_SETTILE, { "G_SETTILE", gfx_set_tile_handler_rdp } },                    // G_SETTILE (-11)
    { RDP_G_FILLRECT, { "G_FILLRECT", gfx_fill_rect_handler_rdp } },                 // G_FILLRECT (-10)
    { RDP_G_SETFILLCOLOR, { "G_SETFILLCOLOR", gfx_set_fill_color_handler_rdp } },    // G_SETFILLCOLOR (-9)
    { RDP_G_SETFOGCOLOR, { "G_SETFOGCOLOR", gfx_set_fog_color_handler_rdp } },       // G_SETFOGCOLOR (-8)
    { RDP_G_SETBLENDCOLOR, { "G_SETBLENDCOLOR", gfx_set_blend_color_handler_rdp } }, // G_SETBLENDCOLOR (-7)
    { RDP_G_SETPRIMCOLOR, { "G_SETPRIMCOLOR", gfx_set_prim_color_handler_rdp } },    // G_SETPRIMCOLOR (-6)
    { RDP_G_SETENVCOLOR, { "G_SETENVCOLOR", gfx_set_env_color_handler_rdp } },       // G_SETENVCOLOR (-5)
    { RDP_G_SETCOMBINE, { "G_SETCOMBINE", gfx_set_combine_handler_rdp } },           // G_SETCOMBINE (-4)
    { RDP_G_SETTIMG, { "G_SETTIMG", gfx_set_timg_handler_rdp } },                    // G_SETTIMG (-3)
    { RDP_G_SETZIMG, { "G_SETZIMG", gfx_set_z_img_handler_rdp } },                   // G_SETZIMG (-2)
    { RDP_G_SETCIMG, { "G_SETCIMG", gfx_set_c_img_handler_rdp } },                   // G_SETCIMG (-1)
};

static constexpr UcodeHandler otrHandlers = {
    { OTR_G_SETTIMG_OTR_HASH,
      { "G_SETTIMG_OTR_HASH", gfx_set_timg_otr_hash_handler_custom } },       // G_SETTIMG_OTR_HASH (0x20)
    { OTR_G_SETFB, { "G_SETFB", gfx_set_fb_handler_custom } },                // G_SETFB (0x21)
    { OTR_G_RESETFB, { "G_RESETFB", gfx_reset_fb_handler_custom } },          // G_RESETFB (0x22)
    { OTR_G_SETTIMG_FB, { "G_SETTIMG_FB", gfx_set_timg_fb_handler_custom } }, // G_SETTIMG_FB (0x23)
    { OTR_G_VTX_OTR_FILEPATH,
      { "G_VTX_OTR_FILEPATH", gfx_vtx_otr_filepath_handler_custom } }, // G_VTX_OTR_FILEPATH (0x24)
    { OTR_G_SETTIMG_OTR_FILEPATH,
      { "G_SETTIMG_OTR_FILEPATH", gfx_set_timg_otr_filepath_handler_custom } }, // G_SETTIMG_OTR_FILEPATH (0x25)
    { OTR_G_TRI1_OTR, { "G_TRI1_OTR", gfx_tri1_otr_handler_f3dex2 } },          // G_TRI1_OTR (0x26)
    { OTR_G_DL_OTR_FILEPATH, { "G_DL_OTR_FILEPATH", gfx_dl_otr_filepath_handler_custom } }, // G_DL_OTR_FILEPATH (0x27)
    { OTR_G_PUSHCD, { "G_PUSHCD", gfx_pushcd_handler_custom } },                            // G_PUSHCD (0x28)
    { OTR_G_MTX_OTR_FILEPATH,
      { "G_MTX_OTR_FILEPATH", gfx_mtx_otr_filepath_handler_custom } },          // G_MTX_OTR_FILEPATH (0x29)
    { OTR_G_DL_OTR_HASH, { "G_DL_OTR_HASH", gfx_dl_otr_hash_handler_custom } }, // G_DL_OTR_HASH (0x31)
    { OTR_G_VTX_OTR_HASH, { "G_VTX_OTR_HASH", gfx_vtx_hash_handler_custom } },  // G_VTX_OTR_HASH (0x32)
    { OTR_G_MARKER, { "G_MARKER", gfx_marker_handler_otr } },                   // G_MARKER (0X33)
    { OTR_G_INVALTEXCACHE, { "G_INVALTEXCACHE", gfx_invalidate_tex_cache_handler_f3dex2 } }, // G_INVALTEXCACHE (0X34)
    { OTR_G_BRANCH_Z_OTR, { "G_BRANCH_Z_OTR", gfx_branch_z_otr_handler_f3dex2 } },           // G_BRANCH_Z_OTR (0x35)
    { OTR_G_MTX_OTR, { "G_MTX_OTR", gfx_mtx_otr_handler_custom } },                          // G_MTX_OTR (0x36)
    { OTR_G_TEXRECT_WIDE, { "G_TEXRECT_WIDE", gfx_tex_rect_wide_handler_custom } },          // G_TEXRECT_WIDE (0x37)
    { OTR_G_FILLWIDERECT, { "G_FILLWIDERECT", gfx_fill_wide_rect_handler_custom } },         // G_FILLWIDERECT (0x38)
    { OTR_G_SETGRAYSCALE, { "G_SETGRAYSCALE", gfx_set_grayscale_handler_custom } },          // G_SETGRAYSCALE (0x39)
    { OTR_G_EXTRAGEOMETRYMODE,
      { "G_EXTRAGEOMETRYMODE", gfx_extra_geometry_mode_handler_custom } }, // G_EXTRAGEOMETRYMODE (0x3a)
    { OTR_G_COPYFB, { "G_COPYFB", gfx_copy_fb_handler_custom } },          // G_COPYFB (0x3b)
    { OTR_G_IMAGERECT, { "G_IMAGERECT", gfx_image_rect_handler_custom } }, // G_IMAGERECT (0x3c)
    { OTR_G_DL_INDEX, { "G_DL_INDEX", gfx_dl_index_handler } },            // G_DL_INDEX (0x3d)
    { OTR_G_READFB, { "G_READFB", gfx_read_fb_handler_custom } },          // G_READFB (0x3e)
    { OTR_G_REGBLENDEDTEX,
      { "G_REGBLENDEDTEX", gfx_register_blended_texture_handler_custom } },         // G_REGBLENDEDTEX (0x3f)
    { OTR_G_SETINTENSITY, { "G_SETINTENSITY", gfx_set_intensity_handler_custom } }, // G_SETINTENSITY (0x40)
    { OTR_G_LINE3D_GDX, { "G_LINE3D_GDX", gfx_line3d_gdx_handler_custom } },        // G_LINE3D_GDX (0x41)
    { OTR_G_MOVEMEM_HASH, { "OTR_G_MOVEMEM_HASH", gfx_movemem_handler_otr } },      // OTR_G_MOVEMEM_HASH
    { OTR_G_PUSH_SHADER, { "G_PUSH_SHADER", gfx_push_shader } },
    { OTR_G_POP_SHADER, { "G_POP_SHADER", gfx_pop_shader } },
    { RDP_G_LOADBLOCK_WIDE, { "G_LOADBLOCK_WIDE", gfx_load_block_wide_handler_rdp } }, // RDP_G_LOADBLOCK_WIDE (-15)
    { RDP_G_VTX_WIDE, { "G_VTX_WIDE", gfx_vtx_handler_f3dex2 } },                      // RDP_G_VTX_WIDE (-16)
    { RDP_G_TRI1_WIDE, { "G_TRI1_WIDE", gfx_tri1_handler_f3dex2 } },                   // RDP_G_TRI1_WIDE (-17)
};

static constexpr UcodeHandler f3dex2Handlers = {
    { F3DEX2_G_NOOP, { "G_NOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX2_G_SPNOOP, { "G_SPNOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX2_G_CULLDL, { "G_CULLDL", gfx_cull_dl_handler_f3dex2 } },
    { F3DEX2_G_MTX, { "G_MTX", gfx_mtx_handler_f3dex2 } },
    { F3DEX2_G_POPMTX, { "G_POPMTX", gfx_pop_mtx_handler_f3dex2 } },
    { F3DEX2_G_MOVEMEM, { "G_MOVEMEM", gfx_movemem_handler_f3dex2 } },
    { F3DEX2_G_DMA_IO, { "G_DMA_IO", gfx_dma_io_handler_f3dex2 } },
    { F3DEX2_G_MOVEWORD, { "G_MOVEWORD", gfx_moveword_handler_f3dex2 } },
    { F3DEX2_G_TEXTURE, { "G_TEXTURE", gfx_texture_handler_f3dex2 } },
    { F3DEX2_G_VTX, { "G_VTX", gfx_vtx_handler_f3dex2 } },
    { F3DEX2_G_MODIFYVTX, { "G_MODIFYVTX", gfx_modify_vtx_handler_f3dex2 } },
    { F3DEX2_G_RDPHALF_1, { "G_RDPHALF_1", gfx_rdphalf_1_handler_f3dex2 } },
    { F3DEX2_G_BRANCH_Z, { "G_BRANCH_Z", gfx_branch_z_handler_f3dex2 } },
    { F3DEX2_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX2_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
    { F3DEX2_G_GEOMETRYMODE, { "G_GEOMETRYMODE", gfx_geometry_mode_handler_f3dex2 } },
    { F3DEX2_G_TRI1, { "G_TRI1", gfx_tri1_handler_f3dex2 } },
    { F3DEX2_G_TRI2, { "G_TRI2", gfx_tri2_handler_f3dex } },
    { F3DEX2_G_QUAD, { "G_QUAD", gfx_quad_handler_f3dex2 } },
    { F3DEX2_G_SETOTHERMODE_L, { "G_SETOTHERMODE_L", gfx_othermode_l_handler_f3dex2 } },
    { F3DEX2_G_SETOTHERMODE_H, { "G_SETOTHERMODE_H", gfx_othermode_h_handler_f3dex2 } },
};

static constexpr UcodeHandler f3dexHandlers = {
    { F3DEX_G_NOOP, { "G_NOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX_G_CULLDL, { "G_CULLDL", gfx_cull_dl_handler_f3dex2 } },
    { F3DEX_G_MTX, { "G_MTX", gfx_mtx_handler_f3d } },
    { F3DEX_G_POPMTX, { "G_POPMTX", gfx_pop_mtx_handler_f3d } },
    { F3DEX_G_MOVEMEM, { "G_POPMEM", gfx_movemem_handler_f3d } },
    { F3DEX_G_MOVEWORD, { "G_MOVEWORD", gfx_moveword_handler_f3d } },
    { F3DEX_G_TEXTURE, { "G_TEXTURE", gfx_texture_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_L, { "G_SETOTHERMODE_L", gfx_othermode_l_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_H, { "G_SETOTHERMODE_H", gfx_othermode_h_handler_f3d } },
    { F3DEX_G_SETGEOMETRYMODE, { "G_SETGEOMETRYMODE", gfx_set_geometry_mode_handler_f3dex } },
    { F3DEX_G_CLEARGEOMETRYMODE, { "G_CLEARGEOMETRYMODE", gfx_clear_geometry_mode_handler_f3dex } },
    { F3DEX_G_VTX, { "G_VTX", gfx_vtx_handler_f3dex } },
    { F3DEX_G_TRI1, { "G_TRI1", gfx_tri1_handler_f3dex } },
    { F3DEX_G_MODIFYVTX, { "G_MODIFYVTX", gfx_modify_vtx_handler_f3dex2 } },
    { F3DEX_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
    { F3DEX_G_TRI2, { "G_TRI2", gfx_tri2_handler_f3dex } },
    { F3DEX_G_SPNOOP, { "G_SPNOOP", gfx_spnoop_command_handler_f3dex2 } },
    { F3DEX_G_RDPHALF_1, { "mRdpHALF_1", gfx_stubbed_command_handler } },
    { F3DEX_G_QUAD, { "G_QUAD", gfx_quad_handler_f3dex } },
};

static constexpr UcodeHandler f3dHandlers = {
    { F3DEX_G_NOOP, { "G_NOOP", gfx_noop_handler_f3dex2 } },
    { F3DEX_G_CULLDL, { "G_CULLDL", gfx_cull_dl_handler_f3dex2 } },
    { F3DEX_G_MTX, { "G_MTX", gfx_mtx_handler_f3d } },
    { F3DEX_G_POPMTX, { "G_POPMTX", gfx_pop_mtx_handler_f3d } },
    { F3DEX_G_MOVEMEM, { "G_POPMEM", gfx_movemem_handler_f3d } },
    { F3DEX_G_MOVEWORD, { "G_MOVEWORD", gfx_moveword_handler_f3d } },
    { F3DEX_G_TEXTURE, { "G_TEXTURE", gfx_texture_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_L, { "G_SETOTHERMODE_L", gfx_othermode_l_handler_f3d } },
    { F3DEX_G_SETOTHERMODE_H, { "G_SETOTHERMODE_H", gfx_othermode_h_handler_f3d } },
    { F3DEX_G_SETGEOMETRYMODE, { "G_SETGEOMETRYMODE", gfx_set_geometry_mode_handler_f3dex } },
    { F3DEX_G_CLEARGEOMETRYMODE, { "G_CLEARGEOMETRYMODE", gfx_clear_geometry_mode_handler_f3dex } },
    { F3DEX_G_VTX, { "G_VTX", gfx_vtx_handler_f3d } },
    { F3DEX_G_TRI1, { "G_TRI1", gfx_tri1_handler_f3d } },
    { F3DEX_G_MODIFYVTX, { "G_MODIFYVTX", gfx_modify_vtx_handler_f3dex2 } },
    { F3DEX_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
    { F3DEX_G_TRI2, { "G_TRI2", gfx_tri2_handler_f3dex } },
    { F3DEX_G_SPNOOP, { "G_SPNOOP", gfx_spnoop_command_handler_f3dex2 } },
    { F3DEX_G_RDPHALF_1, { "mRdpHALF_1", gfx_stubbed_command_handler } },
};

// LUSTODO: These S2DEX commands have different opcode numbers on F3DEX2 vs other ucodes. More research needs to be done
// to see if the implementations are different.
static constexpr UcodeHandler s2dexHandlers = {
    { F3DEX2_G_BG_COPY, { "G_BG_COPY", gfx_bg_copy_handler_s2dex } },
    { F3DEX2_G_BG_1CYC, { "G_BG_1CYC", gfx_bg_1cyc_handler_s2dex } },
    { F3DEX2_G_OBJ_RENDERMODE, { "G_OBJ_RENDERMODE", gfx_stubbed_command_handler } },
    { F3DEX2_G_OBJ_RECTANGLE_R, { "G_OBJ_RECTANGLE_R", gfx_stubbed_command_handler } },
    { F3DEX2_G_OBJ_RECTANGLE, { "G_OBJ_RECTANGLE", gfx_obj_rectangle_handler_s2dex } },
    { F3DEX2_G_DL, { "G_DL", gfx_dl_handler_common } },
    { F3DEX2_G_ENDDL, { "G_ENDDL", gfx_end_dl_handler_common } },
};

static constexpr std::array ucode_handlers = {
    &f3dHandlers,    // ucode_f3db
    &f3dHandlers,    // ucode_f3d
    &f3dexHandlers,  // ucode_f3dex
    &f3dexHandlers,  // ucode_f3dexb
    &f3dex2Handlers, // ucode_f3dex2
    &s2dexHandlers,  // ucode_s2dex
};

const char* GfxGetOpcodeName(int8_t opcode) {
    if (otrHandlers.contains(opcode)) {
        return otrHandlers.at(opcode).first;
    }

    if (rdpHandlers.contains(opcode)) {
        return rdpHandlers.at(opcode).first;
    }

    if (ucode_handler_index < ucode_handlers.size()) {
        if (ucode_handlers[ucode_handler_index]->contains(opcode)) {
            return ucode_handlers[ucode_handler_index]->at(opcode).first;
        } else {
            SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, for loaded ucode: {}", (uint8_t)opcode,
                            (uint32_t)ucode_handler_index);
        }
    } else {
        SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, invalid ucode: {}", (uint8_t)opcode, (uint32_t)ucode_handler_index);
    }

    return nullptr;
}

// TODO, implement a system where we can get the current opcode handler by writing to the GWords. If the powers that be
// are OK with that...
static void gfx_set_ucode_handler(UcodeHandlers ucode) {
    // Loaded ucode must be in range of the supported ucode_handlers
    assert(ucode < ucode_max);
    Interpreter* gfx = mInstance.lock().get();
    const bool handlerChanged = (ucode_handler_index != ucode);
    ucode_handler_index = ucode;

    // Same handler table means an in-family variant switch (F-Zero X issues G_LOAD_UCODE from
    // F3DEX2 to F3DFLX2.Rej mid-frame, before every machine). Hardware does not clear RSP
    // constants on such a load, and the game relies on Course_Draw's fog position staying live
    // for the car draws that follow, so apply the quirk resets only on a real table change.
    if (!handlerChanged) {
        return;
    }

    // Reset some RSP state values upon ucode load to deal with hardware quirks discovered by emulators
    switch (ucode) {
        case ucode_f3d:
        case ucode_f3db:
        case ucode_f3dex:
        case ucode_f3dexb:
        case ucode_f3dex2:
            gfx->mRsp->fog_mul = 0;
            gfx->mRsp->fog_offset = 0;
            break;
        default:
            break;
    }
}

static void gfx_step() {
    auto& cmd = g_exec_stack.currCmd();
    auto cmd0 = cmd;
    int8_t opcode = (int8_t)(cmd->words.w0 >> 24);

#ifdef USE_GBI_TRACE
    if (cmd->words.trace.valid &&
        Ship::Context::GetInstance()->GetConsoleVariables()->GetInteger("gEnableGFXTrace", 0)) {
#define TRACE                                  \
    "\n====================================\n" \
    " - CMD: {:02X}\n"                         \
    " - Path: {}:{}\n"                         \
    " - W0: {:08X}\n"                          \
    " - W1: {:08X}\n"                          \
    "===================================="
        SPDLOG_INFO(TRACE, (uint8_t)opcode, cmd->words.trace.file, cmd->words.trace.idx, cmd->words.w0, cmd->words.w1);
    }
#endif

    if (opcode == F3DEX2_G_LOAD_UCODE) {
        const uintptr_t payload = static_cast<uintptr_t>(cmd->words.w1);
        if ((payload & ~static_cast<uintptr_t>(0xFFu)) == F3DEX2_VARIANT_SWITCH_MARKER) {
            const uint8_t rawVariant = static_cast<uint8_t>(payload & 0xFFu);
            if (rawVariant <= static_cast<uint8_t>(F3dex2Variant::FZeroFlxReject)) {
                gfx_set_ucode_handler(ucode_f3dex2);
                mInstance.lock()->SetF3dex2Variant(static_cast<F3dex2Variant>(rawVariant));
            }
        } else {
            gfx_set_ucode_handler((UcodeHandlers)(cmd->words.w0 & 0xFFFFFF));
        }
        ++cmd;
        return;
        // Instead of having a handler for each ucode for switching ucode, just check for it early and return.
    }

    if (otrHandlers.contains(opcode)) {
        // OTR filepath handlers expect w1 to be a valid string pointer.
        // Guard against null or N64-segment addresses that would crash in strlen/strncmp.
        if (opcode == OTR_G_VTX_OTR_FILEPATH || opcode == OTR_G_SETTIMG_OTR_FILEPATH ||
            opcode == OTR_G_DL_OTR_FILEPATH || opcode == OTR_G_PUSHCD || opcode == OTR_G_MTX_OTR_FILEPATH) {
            uintptr_t w1 = (uintptr_t)cmd->words.w1;
            if (w1 < 0x10000
#if UINTPTR_MAX > 0xFFFFFFFFu
                // On 64-bit: filter kernel/sentinel addresses.
                || w1 > 0x0000FFFFFFFFFFFFull
#endif
            ) {
                ++g_exec_stack.currCmd();
                return;
            }
        }
        if (otrHandlers.at(opcode).second(&cmd)) {
            return;
        }
    } else if (rdpHandlers.contains(opcode)) {
        if (rdpHandlers.at(opcode).second(&cmd)) {
            return;
        }
    } else if (ucode_handler_index < ucode_handlers.size()) {
        if (ucode_handlers[ucode_handler_index]->contains(opcode)) {
            if (ucode_handlers[ucode_handler_index]->at(opcode).second(&cmd)) {
                return;
            }
        } else {
            SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, for loaded ucode: {}", (uint8_t)opcode,
                            (uint32_t)ucode_handler_index);
        }
    } else {
        SPDLOG_CRITICAL("Unhandled OP code: 0x{:X}, invalid ucode: {}", (uint8_t)opcode, (uint32_t)ucode_handler_index);
    }

    ++cmd;
}

void Interpreter::SpReset() {
    while (!mShaderStack.empty()) {
        mShaderStack.pop();
    }
    // A widescreen anchor/stretch scope set by a display list that was branched over or
    // aborted mid-frame must not leak into the next task: a latched G_EX_WIDESCREEN_STRETCH
    // would mis-stretch every subsequent 2D rect. The game clears its scopes explicitly on
    // every normal path; this is the per-task backstop.
    mRsp->extra_geometry_mode = 0;
    mRsp->modelview_matrix_stack_size = 1;
    mRsp->branch_z_target = 0;
    mRsp->viewport_z_scale = 511.0f;
    mRsp->viewport_z_trans = 511.0f;
    mRsp->current_num_lights = 2;
    mRsp->lights_changed = true;
    memset(mRsp->dmem, 0, sizeof(mRsp->dmem));
    mRsp->dma_io_dmem = 0;
    mRsp->dma_io_loaded = false;
    mRsp->f3dflx_alpha_light_valid = false;
    mRsp->lookat[0].dir[0] = 0;
    mRsp->lookat[0].dir[1] = 127;
    mRsp->lookat[0].dir[2] = 0;
    mRsp->lookat[1].dir[0] = 127;
    mRsp->lookat[1].dir[1] = 0;
    mRsp->lookat[1].dir[2] = 0;
    CalculateNormalDir(&mRsp->lookat[0], mRsp->current_lookat_coeffs[0]);
    CalculateNormalDir(&mRsp->lookat[1], mRsp->current_lookat_coeffs[1]);
}

// [interp-idem] Hermetic replay baseline for frame-interpolation sub-frame passes (k > 0).
// SpReset() already covers the RSP subset; this zeroes the RDP side that survives Run() --
// TMEM and its generation counter, per-word load bookkeeping, palette staging, tile
// descriptors, rasterizer modes, color registers, chroma-key state -- plus the renderer's
// tracked binding state so the first command of the replay re-applies everything. Starting
// from a clean DEFAULT state (rather than rewinding to pass 0's start) is deliberate: it keeps
// loaded_texture[] free of stale resource handles and keeps the emulated TMEM/generation pair
// in sync with the content-keyed GPU texture cache, which re-executed loads hit instead of
// re-uploading. The GPU cache itself, framebuffer attachments, and mFbActive are untouched:
// StartFrame() re-establishes them at the top of every DrawAndRunGraphicsCommands pass.
void Interpreter::ResetRdpForReplay() {
    if (mRsp != nullptr) {
        // geometry_mode carries G_CULL_BACK/FRONT, G_LIGHTING, G_FOG, G_TEXTURE_GEN and is read
        // for the cull test; SpReset() deliberately does not clear it because a task may rely on
        // mode set by an earlier task in the same frame. A replay must not inherit pass 0's END
        // state, so the replay baseline clears it here.
        mRsp->geometry_mode = 0;
    }
    if (mRdp == nullptr) {
        return;
    }
    std::memset(mRdp->tmem, 0, sizeof(mRdp->tmem));
    mRdp->tmem_generation = 0;
    for (auto& lt : mRdp->loaded_texture) {
        lt = {};
    }
    std::memset(mRdp->palette_staging, 0, sizeof(mRdp->palette_staging));
    mRdp->palettes[0] = mRdp->palettes[1] = nullptr;
    mRdp->palette_dram_addr[0] = mRdp->palette_dram_addr[1] = nullptr;
    mRdp->texture_to_load = {};
    for (auto& tile : mRdp->texture_tile) {
        tile = {};
    }
    mRdp->textures_changed[0] = mRdp->textures_changed[1] = false;
    mRdp->first_tile_index = 0;
    mRdp->other_mode_l = 0;
    mRdp->other_mode_h = 0;
    mRdp->combine_mode = 0;
    mRdp->grayscale = false;
    mRdp->prim_lod_fraction = 0;
    mRdp->prim_depth = 0;
    mRdp->env_color = {};
    mRdp->prim_color = {};
    mRdp->fog_color = {};
    mRdp->blend_color = {};
    mRdp->fill_color = {};
    mRdp->grayscale_color = {};
    mRdp->key_center = {};
    mRdp->key_scale = {};
    for (int16_t& k : mRdp->convert_k) {
        k = 0;
    }
    mRdp->viewport = {};
    mRdp->scissor = {};
    mRdp->viewport_or_scissor_changed = true;

    // Renderer-side tracked state: drop cached texture-node bindings and sampler validity so a
    // replay cannot reuse pass 0's "currently bound" knowledge after its emulated inputs were
    // just zeroed. Cache entries themselves stay alive and uploaded; only the tracking resets.
    std::fill(std::begin(mRenderingState.mTextures), std::end(mRenderingState.mTextures), nullptr);
    std::fill(std::begin(mRenderingState.sampler_valid), std::end(mRenderingState.sampler_valid), false);
}

void Interpreter::RegisterFbTexture(const void* cpuAddr, int fbId) {
    mFbTextures[(uintptr_t)cpuAddr] = fbId;
}

void Interpreter::UnregisterFbTexture(const void* cpuAddr) {
    mFbTextures.erase((uintptr_t)cpuAddr);
}

void Interpreter::GetDimensions(uint32_t* width, uint32_t* height, int32_t* posX, int32_t* posY) {
    mWapi->GetDimensions(width, height, posX, posY);
}

void Interpreter::Init(class GfxWindowBackend* wapi, class GfxRenderingAPI* rapi, const char* game_name,
                       bool start_in_fullscreen, uint32_t width, uint32_t height, uint32_t posX, uint32_t posY) {
    mWapi = wapi;
    mRapi = rapi;
    mWapi->Init(game_name, rapi->GetName(), start_in_fullscreen, width, height, posX, posY);
    mRapi->Init();
    mRapi->UpdateFramebufferParameters(0, width, height, 1, false, true, true, true);
    mCurDimensions.internal_mul =
        Ship::Context::GetInstance()->GetConsoleVariables()->GetFloat(CVAR_INTERNAL_RESOLUTION, 1);
    mMsaaLevel = Ship::Context::GetInstance()->GetConsoleVariables()->GetInteger(CVAR_MSAA_VALUE, 1);

    mCurDimensions.width = width;
    mCurDimensions.height = height;

    mGameFb = mRapi->CreateFramebuffer();
    mGameFbMsaaResolved = mRapi->CreateFramebuffer();

    mNativeDimensions.width = SCREEN_WIDTH;
    mNativeDimensions.height = SCREEN_HEIGHT;

    for (int i = 0; i < MAX_SEGMENT_POINTERS; i++) {
        mSegmentPointers[i] = 0;
    }

    if (mTexUploadBuffer == nullptr) {
        // We cap texture max to 8k, because why would you need more?
        int max_tex_size = std::min(8192, mRapi->GetMaxTextureSize());
        mTexUploadBuffer = (uint8_t*)malloc(max_tex_size * max_tex_size * 4);
    }

    ucode_handler_index = UcodeHandlers::ucode_f3dex2;

    // Pre-allocate texture cache buckets to prevent rehash-induced iterator invalidation.
    mTextureCache.map.reserve(TEXTURE_CACHE_MAX_SIZE);
}

void Interpreter::Destroy() {
    // TODO: should also destroy rapi, and any other resources acquired in fast3d
    free(mTexUploadBuffer);
    mWapi->Destroy();

    // Texture cache and loaded textures store references to Resources which need to be unreferenced.
    TextureCacheClear();
    mRdp->texture_to_load.raw_tex_metadata.resource = nullptr;
    for (auto& loadedTexture : mRdp->loaded_texture) {
        loadedTexture.raw_tex_metadata.resource = nullptr;
    }
}

GfxRenderingAPI* Interpreter::GetCurrentRenderingAPI() {
    return mRapi;
}

void Interpreter::SetGfxDebugger(std::shared_ptr<GfxDebugger> debugger) {
    mGfxDebugger = std::move(debugger);
}

std::shared_ptr<GfxDebugger> Interpreter::GetGfxDebugger() const {
    return mGfxDebugger;
}

void Interpreter::HandleWindowEvents() {
    mWapi->HandleEvents();
}

bool Interpreter::IsFrameReady() {
    return mWapi->IsFrameReady();
}

bool Interpreter::ViewportMatchesRendererResolution() {
#ifdef __APPLE__
    // Always treat the viewport as not matching the render resolution on mac
    // to avoid issues with retina scaling.
    return false;
#else
    if (mCurDimensions.width == mGameWindowViewport.width && mCurDimensions.height == mGameWindowViewport.height) {
        return true;
    }
    return false;
#endif
}

// Whether StartFrame sized mGameFbMsaaResolved this frame. Run() re-latches the fixed-aspect flag
// mid-frame, so a mode flip after StartFrame can request an MSAA resolve into a target StartFrame
// never allocated; the epilogue then resolves to 0 instead of publishing an unallocated buffer.
static bool sGameFbMsaaResolvedSized = false;

void Interpreter::StartFrame() {
    mWapi->GetDimensions(&mGfxCurrentWindowDimensions.width, &mGfxCurrentWindowDimensions.height, &mCurWindowPosX,
                         &mCurWindowPosY);
    if (mCurDimensions.height == 0) {
        // Avoid division by zero
        mCurDimensions.height = 1;
    }
    mCurDimensions.aspect_ratio = (float)mCurDimensions.width / (float)mCurDimensions.height;

    // Update the framebuffer sizes when the viewport or native dimension changes
    if (mCurDimensions.width != mPrvDimensions.width || mCurDimensions.height != mPrvDimensions.height ||
        mNativeDimensions.width != mPrevNativeDimensions.width ||
        mNativeDimensions.height != mPrevNativeDimensions.height) {

        for (auto& fb : mFrameBuffers) {
            uint32_t width = fb.second.orig_width, height = fb.second.orig_height;
            if (fb.second.resize) {
                AdjustWidthHeightForScale(width, height, fb.second.native_width, fb.second.native_height);
            }
            if (width != fb.second.applied_width || height != fb.second.applied_height) {
                mRapi->UpdateFramebufferParameters(fb.first, width, height, 1, true, true, true, true);
                fb.second.applied_width = width;
                fb.second.applied_height = height;
            }
        }
    }

    mPrvDimensions = mCurDimensions;
    mPrevNativeDimensions = mNativeDimensions;
    // Latch the widescreen CVars once per frame; the per-vertex and per-rect consumers read these
    // members instead of doing CVar lookups on the hot path. Run() re-latches the fixed-aspect
    // flag per task. Widescreen == 0 pillarboxes via Fast3dGui::DrawGame, which needs the frame in
    // an offscreen texture -- hence the forced offscreen target below even at 1x with no MSAA.
    mWidescreenEnabledCache = CVarGetInteger("gEnhancements.Graphics.Widescreen", 1) != 0;
    mForceFixedAspectCache = sGdxForceFixedAspect != 0;
    mWidescreenUiCache = CVarGetInteger("gEnhancements.Graphics.WidescreenUI", 0) != 0;
    mUltrawideCache = CVarGetInteger("gEnhancements.Graphics.UltrawideMode", 0) != 0;
    mRemoveBordersCache = CVarGetInteger("gEnhancements.Graphics.RemoveBorders", 0) != 0;
    mPostShaderCache = CVarGetInteger("gEnhancements.Graphics.CRTShader", 0);
    // A selected custom shader must also force the offscreen path; the mode value stored here is
    // only passed to the backend for built-in modes, while the backend reads the CustomShader CVar
    // directly when it is non-empty.
    if (mPostShaderCache == 0 && CVarGetString("gEnhancements.Graphics.CustomShader", "")[0] != '\0') {
        mPostShaderCache = 1;
    }
    // An active .slangp pipeline or loose .slang shader forces the offscreen path too. The
    // preset is parsed here so a compile or parse error is logged once and the frame falls back
    // to the unfiltered source.
    const char* postPipelineCVar = CVarGetString("gEnhancements.Graphics.PostPipeline", "");
    mPostPipelineCache = postPipelineCVar != nullptr ? postPipelineCVar : "";
    // Tracing a stored selection that reverts across runs; temporary for 1.1.0 RC.
    static std::string sLastLoggedPostPipeline = "\x01";
    if (mPostPipelineCache != sLastLoggedPostPipeline) {
        SPDLOG_INFO("PostPipeline latched '{}' (was '{}')", mPostPipelineCache,
                    sLastLoggedPostPipeline == "\x01" ? "<unset>" : sLastLoggedPostPipeline);
        sLastLoggedPostPipeline = mPostPipelineCache;
    }
    const int builtinPostMode = mPostShaderCache;
    if (!mPostPipelineCache.empty()) {
        mPostShaderCache = 1;
        const std::filesystem::path appDir = Ship::Context::GetAppDirectoryPath();
        const bool isLooseSlang =
            mPostPipelineCache.size() > 6 && mPostPipelineCache.compare(mPostPipelineCache.size() - 6, 6, ".slang") == 0;
        const std::filesystem::path presetPath = GdxResolvePostShaderPath(appDir, mPostPipelineCache);
        bool needParse = true;
        if (mPostPipeline != nullptr && mPostPipeline->presetPath == presetPath) {
            needParse = GdxPostShaderPipelineChanged(*mPostPipeline);
        }
        if (mPostPipeline == nullptr && mPostPipelineFailedPath == presetPath) {
            // Previously failed; only retry when the file was touched since.
            needParse = GdxGetMtime(presetPath) != mPostPipelineFailedMtime;
        }
        if (needParse) {
            auto newPipeline = std::make_unique<GdxPostShaderPipeline>();
            std::string error;
            bool ok = false;
            if (isLooseSlang) {
                // A loose .slang file is treated as a one-pass pipeline at source scale.
                if (std::filesystem::is_regular_file(presetPath)) {
                    GdxPostPassDesc pass;
                    pass.shader = presetPath.stem().string();
                    newPipeline->presetPath = presetPath;
                    newPipeline->mtime = GdxGetMtime(presetPath);
                    newPipeline->passes.push_back(std::move(pass));
                    ok = true;
                } else {
                    error = "Loose slang shader not found: " + presetPath.string();
                }
            } else {
                // A pathological preset must not take the game down; a failed parse
                // is cached below and the frame renders unfiltered.
                try {
                    ok = GdxParsePostShaderPipeline(presetPath, newPipeline.get(), &error);
                } catch (const std::exception& e) {
                    error = e.what();
                } catch (...) {
                    error = "unknown exception";
                }
            }
            if (ok) {
                mPostPipeline = std::move(newPipeline);
                mPostPipelineFailedPath.clear();
            } else {
                mPostPipeline.reset();
                mPostPipelineFailedPath = presetPath.generic_string();
                mPostPipelineFailedMtime = GdxGetMtime(presetPath);
                SPDLOG_ERROR("Failed to parse post pipeline '{}': {}", mPostPipelineCache, error);
            }
        }
        // A preset that failed to load — fresh parse failure or a cached one skipped above —
        // must not keep the forced built-in mode; restore the built-in selector's mode.
        if (mPostPipeline == nullptr) {
            mPostShaderCache = builtinPostMode;
        }
    } else {
        mPostPipeline.reset();
        mPostPipelineFailedPath.clear();
    }
    // HudMaxAspect: 0 or any out-of-range value means no clamp, so ANCHOR-scoped HUD elements
    // glue to the true screen corners. A value in [4:3, 8) confines the HUD to a centred band of
    // that aspect.
    mHudMaxAspectCache = CVarGetFloat("gEnhancements.Graphics.HudMaxAspect", 0.0f);
    if (!(mHudMaxAspectCache >= 4.0f / 3.0f && mHudMaxAspectCache < 8.0f)) {
        mHudMaxAspectCache = 1000.0f; // min(currentAspect, 1000) == currentAspect -> old edge-glue math
    }
    const bool widescreenPillarbox = !mWidescreenEnabledCache || mForceFixedAspectCache;
    sGameFbMsaaResolvedSized = false;
    if (mVrEyeActive && mVrPass == VrPass::HudOnly) {
        // The VR HUD layer is a wide panel: lay the HUD out with the widescreen UI logic so the
        // edge-anchored elements reach the panel's sides while central ones keep 4:3 proportions,
        // whatever the desktop widescreen settings are.
        mWidescreenEnabledCache = true;
        mWidescreenUiCache = true;
        mForceFixedAspectCache = false;
        mHudMaxAspectCache = 1000.0f;
    }
    if (mVrEyeActive) {
        // Eye frames are read back as a texture by the VR presenter: always offscreen, exactly
        // eye-sized, single-sampled, unfiltered.
        mPostShaderCache = 0;
        mRendersToFb = true;
        mRapi->UpdateFramebufferParameters(mGameFb, mCurDimensions.width, mCurDimensions.height, 1, true, true, true,
                                           true);
    } else if (!ViewportMatchesRendererResolution() || mMsaaLevel > 1 || widescreenPillarbox ||
               mPostShaderCache != 0) {
        mRendersToFb = true;
        if (!ViewportMatchesRendererResolution() || (widescreenPillarbox && mMsaaLevel <= 1)) {
            mRapi->UpdateFramebufferParameters(mGameFb, mCurDimensions.width, mCurDimensions.height, mMsaaLevel, true,
                                               true, true, true);
        } else {
            // MSAA framebuffer needs to be resolved to an equally sized target when complete, which must therefore
            // match the window size
            mRapi->UpdateFramebufferParameters(mGameFb, mGfxCurrentWindowDimensions.width,
                                               mGfxCurrentWindowDimensions.height, mMsaaLevel, false, true, true, true);
        }
        // The MSAA resolve target must also exist when a whole-frame pillarbox is required: the
        // epilogue resolves into it and publishes it as mGfxFrameBuffer so DrawGame composites the
        // centred 4:3 region. Resolving straight to the window would stretch 4:3 content across it.
        // The post shader needs it too: with a post mode on there is no direct-to-window resolve,
        // because the window backbuffer has no texture for the post pass to sample.
        if (mMsaaLevel > 1 && (!ViewportMatchesRendererResolution() || widescreenPillarbox || mPostShaderCache != 0)) {
            // Store the resolve target top-down so DrawGame's ImGui::Image path presents it the
            // same way as the non-MSAA offscreen framebuffer (mGameFb is also top-down in these
            // branches). ResolveMSAAColorBuffer now flips when src/dst invertY differ.
            mRapi->UpdateFramebufferParameters(mGameFbMsaaResolved, mCurDimensions.width, mCurDimensions.height, 1,
                                               true, false, false, false);
            sGameFbMsaaResolvedSized = true;
        }
    } else {
        mRendersToFb = false;
    }

    mFbActive = false;
}

GfxExecStack g_exec_stack = {};

void Interpreter::RunGuiOnly() {
    SpReset();

    // Unlike Run(), this path does not re-latch mForceFixedAspectCache, so there is no staleness
    // gap (mode flips arrive with a game task, via Run()). If a re-latch is ever added here, it
    // must also recompute mRendersToFb the way Run()'s prologue does.

    mGetPixelDepthPending.clear();
    mGetPixelDepthCached.clear();

    mRapi->UpdateFramebufferParameters(0, mGfxCurrentWindowDimensions.width, mGfxCurrentWindowDimensions.height, 1,
                                       false, true, true, !mRendersToFb);
    mRapi->StartFrame();
    mRapi->StartDrawToFramebuffer(mRendersToFb ? mGameFb : 0, (float)mCurDimensions.height / mNativeDimensions.height);
    mRapi->ClearFramebuffer(true, true);
    mRdp->viewport_or_scissor_changed = true;
    mRenderingState.viewport = {};
    mRenderingState.scissor = {};

    Flush();
    mGfxFrameBuffer = 0;

    if (mRendersToFb) {
        const bool widescreenPillarbox = !mWidescreenEnabledCache || mForceFixedAspectCache;
        mRapi->StartDrawToFramebuffer(0, 1);
        mRapi->ClearFramebuffer(true, true);
        int postSrcFb = -1;
        if (mMsaaLevel > 1) {
            if (ViewportMatchesRendererResolution() && !widescreenPillarbox && mPostShaderCache == 0) {
                // Normal path: resolve straight to the window; mGfxFrameBuffer stays
                // 0 and DrawGame presents it directly.
                mRapi->ResolveMSAAColorBuffer(0, mGameFb);
            } else if (sGameFbMsaaResolvedSized) {
                // Resolve into the offscreen target and publish it so DrawGame
                // composites (pillarboxes) it instead of stretching.
                mRapi->ResolveMSAAColorBuffer(mGameFbMsaaResolved, mGameFb);
                mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFbMsaaResolved);
                postSrcFb = mGameFbMsaaResolved;
            } else {
                // A config flip after StartFrame requested the offscreen resolve, but
                // mGameFbMsaaResolved was never sized this frame; resolve to 0 rather
                // than publish an unallocated framebuffer's texture id.
                mRapi->ResolveMSAAColorBuffer(0, mGameFb);
            }
        } else {
            mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFb);
            postSrcFb = mGameFb;
        }
        if (mPostShaderCache != 0 && postSrcFb >= 0) {
            // The post target matches the source framebuffer's size, which StartFrame sized to
            // mCurDimensions for offscreen-ratio renders and to the window otherwise. DrawGame's
            // rect math reads only mCurDimensions' aspect, so either size composites the same.
            uint32_t postOutW, postOutH;
            if (postSrcFb == mGameFbMsaaResolved || !ViewportMatchesRendererResolution() || widescreenPillarbox) {
                postOutW = mCurDimensions.width;
                postOutH = mCurDimensions.height;
            } else {
                postOutW = mGfxCurrentWindowDimensions.width;
                postOutH = mGfxCurrentWindowDimensions.height;
            }
            uintptr_t postTex = 0;
            if (mPostPipeline != nullptr) {
                postTex = mRapi->ApplyPostShaderChain(postSrcFb, *mPostPipeline, mNativeDimensions.width,
                                                      mNativeDimensions.height, postOutW, postOutH);
            } else {
                postTex = mRapi->ApplyPostShader(postSrcFb, mPostShaderCache, mNativeDimensions.width,
                                                 mNativeDimensions.height, postOutW, postOutH);
            }
            // 0 means the backend declined (unsupported or compile failed); keep the
            // unfiltered source published rather than break presentation.
            if (postTex != 0) {
                mGfxFrameBuffer = postTex;
            }
        }
    } else if (mFbActive) {
        // Failsafe reset to main framebuffer to prevent softlocking the renderer
        mFbActive = 0;
        mRapi->StartDrawToFramebuffer(0, 1);

        assert(0 && "active framebuffer was never reset back to original");
    }
}

void (*Interpreter::sPortAfterClearHook)(Interpreter*) = nullptr;

void Interpreter::SetPortAfterClearHook(void (*hook)(Interpreter*)) {
    sPortAfterClearHook = hook;
}

void Interpreter::Run(Gfx* commands, const std::unordered_map<Mtx*, MtxF>& mtx_replacements) {
    SpReset();
    mVrStopRequested = false;
    mVrMarkerSeen = false;
    mVrInHudSprites = false;
    mVrWorldSpriteActive = false;
    mVrSuppressDraw = (mVrPass == VrPass::HudOnly);
    gGdxVrWorldSpritesLastRun = 0;
    gGdxVrHudSpritesLastRun = 0;

    // Re-latch the fixed-aspect flag per task: the game can flip its mode between this frame's
    // StartFrame and the gfx task that renders the new mode, so the StartFrame latch alone applies
    // the old mode's aspect to the new mode's first frame -- a 4:3 squeeze on editor exit.
    mForceFixedAspectCache = sGdxForceFixedAspect != 0;

    // Recompute the render-target decision with the fresh flag: StartFrame latched mRendersToFb
    // before this frame's dispatch, and a mid-dispatch mode flip would otherwise leave vertices on
    // the new mode while rendering to the old target (one squashed, un-pillarboxed frame).
    // Prologue and epilogue both read the mRendersToFb set here. Only mMsaaLevel <= 1 can flip it,
    // since MSAA forces it true in StartFrame; a flip that stales the MSAA resolve sizing is
    // caught by the epilogue's fallback branch.
    {
        const bool widescreenPillarbox = !mWidescreenEnabledCache || mForceFixedAspectCache;
        const bool rendersToFb = mVrEyeActive || !ViewportMatchesRendererResolution() || mMsaaLevel > 1 ||
                                 widescreenPillarbox || mPostShaderCache != 0;
        if (rendersToFb && !mRendersToFb && mMsaaLevel <= 1) {
            // StartFrame never sized mGameFb this frame; reproduce its non-MSAA
            // sizing so the freshly-targeted offscreen render matches the window.
            mRapi->UpdateFramebufferParameters(mGameFb, mCurDimensions.width, mCurDimensions.height, mMsaaLevel, true,
                                               true, true, true);
        }
        mRendersToFb = rendersToFb;
    }

    mGetPixelDepthPending.clear();
    mGetPixelDepthCached.clear();

    mCurMtxReplacements = &mtx_replacements;

    mRapi->UpdateFramebufferParameters(0, mGfxCurrentWindowDimensions.width, mGfxCurrentWindowDimensions.height, 1,
                                       false, true, true, !mRendersToFb);
    mRapi->StartFrame();
    mRapi->StartDrawToFramebuffer(mRendersToFb ? mGameFb : 0, (float)mCurDimensions.height / mNativeDimensions.height);
    mRapi->ClearFramebuffer(true, true);
    mRdp->viewport_or_scissor_changed = true;
    mRenderingState.viewport = {};
    mRenderingState.scissor = {};

    // Runs on the freshly-cleared canvas, before the task's commands, so its content (the
    // boot-logo CPU framebuffer) draws as a background under whatever this task renders.
    if (sPortAfterClearHook != nullptr) {
        sPortAfterClearHook(this);
        // The seed draws a full-frame copy-mode rectangle, which dirties the tracked
        // viewport/scissor state; reset so the task's first command re-applies its own.
        mRdp->viewport_or_scissor_changed = true;
        mRenderingState.viewport = {};
        mRenderingState.scissor = {};
    }

    auto dbg = mGfxDebugger;
    g_exec_stack.start((F3DGfx*)commands);
    while (!g_exec_stack.cmd_stack.empty()) {
        if (mVrStopRequested) {
            g_exec_stack.stop();
            break;
        }
        auto cmd = g_exec_stack.cmd_stack.top();

        if (dbg->IsDebugging()) {
            g_exec_stack.gfx_path.push_back(cmd);
            if (dbg->HasBreakPoint(g_exec_stack.gfx_path)) {
                // On a breakpoint with the active framebuffer still set, we need to reset back to prevent
                // soft locking the renderer
                if (mFbActive) {
                    mFbActive = 0;
                    mRapi->StartDrawToFramebuffer(mRendersToFb ? mGameFb : 0, 1);
                }

                break;
            }
            g_exec_stack.gfx_path.pop_back();
        }
        gfx_step();
    }

    Flush();

    mGfxFrameBuffer = 0;
    currentDir = std::stack<std::string>();

    if (mRendersToFb) {
        // Recompute from the caches as they stand at task end: the prologue re-latch
        // means this reflects the mode actually rendered, not StartFrame's latch.
        const bool widescreenPillarbox = !mWidescreenEnabledCache || mForceFixedAspectCache;
        mRapi->StartDrawToFramebuffer(0, 1);
        mRapi->ClearFramebuffer(true, true);
        int postSrcFb = -1;
        if (mMsaaLevel > 1) {
            if (ViewportMatchesRendererResolution() && !widescreenPillarbox && mPostShaderCache == 0) {
                // Normal path: resolve straight to the window; mGfxFrameBuffer stays
                // 0 and DrawGame presents it directly.
                mRapi->ResolveMSAAColorBuffer(0, mGameFb);
            } else if (sGameFbMsaaResolvedSized) {
                // Resolve into the offscreen target and publish it so DrawGame
                // composites the centred 4:3 region (gated on mGfxFrameBuffer != 0).
                mRapi->ResolveMSAAColorBuffer(mGameFbMsaaResolved, mGameFb);
                mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFbMsaaResolved);
                postSrcFb = mGameFbMsaaResolved;
            } else {
                // The mode flipped to pillarbox after StartFrame, so the offscreen
                // target is not allocated this frame. Resolve to 0 for one frame
                // rather than publish an unallocated framebuffer; the next
                // StartFrame sizes it.
                mRapi->ResolveMSAAColorBuffer(0, mGameFb);
            }
        } else {
            mGfxFrameBuffer = (uintptr_t)mRapi->GetFramebufferTextureId(mGameFb);
            postSrcFb = mGameFb;
        }
        if (mPostShaderCache != 0 && postSrcFb >= 0) {
            // The post target matches the source framebuffer's size, which StartFrame sized to
            // mCurDimensions for offscreen-ratio renders and to the window otherwise. DrawGame's
            // rect math reads only mCurDimensions' aspect, so either size composites the same.
            uint32_t postOutW, postOutH;
            if (postSrcFb == mGameFbMsaaResolved || !ViewportMatchesRendererResolution() || widescreenPillarbox) {
                postOutW = mCurDimensions.width;
                postOutH = mCurDimensions.height;
            } else {
                postOutW = mGfxCurrentWindowDimensions.width;
                postOutH = mGfxCurrentWindowDimensions.height;
            }
            uintptr_t postTex = 0;
            if (mPostPipeline != nullptr) {
                postTex = mRapi->ApplyPostShaderChain(postSrcFb, *mPostPipeline, mNativeDimensions.width,
                                                      mNativeDimensions.height, postOutW, postOutH);
            } else {
                postTex = mRapi->ApplyPostShader(postSrcFb, mPostShaderCache, mNativeDimensions.width,
                                                 mNativeDimensions.height, postOutW, postOutH);
            }
            // 0 means the backend declined (unsupported or compile failed); keep the
            // unfiltered source published rather than break presentation.
            if (postTex != 0) {
                mGfxFrameBuffer = postTex;
            }
        }
    } else if (mFbActive) {
        // Failsafe reset to main framebuffer to prevent softlocking the renderer
        mFbActive = 0;
        mRapi->StartDrawToFramebuffer(0, 1);

        assert(0 && "active framebuffer was never reset back to original");
    }
}

void Interpreter::OnVrHudMarker() {
    mVrMarkerSeen = true;
    if (mVrPass == VrPass::WorldOnly) {
        mVrStopRequested = true; // Run() stops before the next command
    } else if (mVrPass == VrPass::HudOnly) {
        mVrSuppressDraw = false;
    }
}

bool (*Interpreter::sVrWorldSpriteResolver)(uint32_t id, VrWorldSprite* out) = nullptr;

// Per-Run counts of VR sprites actually drawn, for the host's diagnostics.
extern "C" {
int gGdxVrWorldSpritesLastRun = 0;
int gGdxVrHudSpritesLastRun = 0;
}

void Interpreter::SetVrWorldSpriteResolver(bool (*resolver)(uint32_t id, VrWorldSprite* out)) {
    sVrWorldSpriteResolver = resolver;
}

void Interpreter::OnVrSpriteMarker(uint32_t code, uint32_t id) {
    switch (code) {
        case kVrHudSpritesBegin:
            mVrInHudSprites = true;
            break;
        case kVrHudSpritesEnd:
            mVrInHudSprites = false;
            break;
        case kVrWorldSpriteBegin:
            mVrWorldSpriteActive = sVrWorldSpriteResolver != nullptr && sVrWorldSpriteResolver(id, &mVrWorldSprite);
            break;
        case kVrWorldSpriteEnd:
            mVrWorldSpriteActive = false;
            break;
        default:
            break;
    }
}

// Re-place a world-anchored sprite for the eye being rendered: project its anchor with the loaded
// projection (the eye's projection*view; world sprites are drawn with an identity modelview) and
// rebuild the whole rectangle around that point, since the game's rectangle was clipped against
// the game camera's screen. Also moves the primitive depth to the anchor's eye depth. Returns false
// when the anchor is behind the eye.
bool Interpreter::VrPlaceWorldSprite(int32_t* ulx, int32_t* uly, int32_t* lrx, int32_t* lry, int16_t* uls,
                                     int16_t* ult, int16_t* dsdx, int16_t* dtdy) {
    const VrWorldSprite& ws = mVrWorldSprite;
    const float(*P)[4] = mRsp->P_matrix;
    const float cx = ws.x * P[0][0] + ws.y * P[1][0] + ws.z * P[2][0] + P[3][0];
    const float cy = ws.x * P[0][1] + ws.y * P[1][1] + ws.z * P[2][1] + P[3][1];
    const float cz = ws.x * P[0][2] + ws.y * P[1][2] + ws.z * P[2][2] + P[3][2];
    const float cw = ws.x * P[0][3] + ws.y * P[1][3] + ws.z * P[2][3] + P[3][3];
    if (!(cw > 0.001f)) {
        return false;
    }
    const float sx = 160.0f + (cx / cw) * 160.0f;
    const float sy = 120.0f - (cy / cw) * 120.0f;
    // The sprite was sized in 320x240 pixels of the game camera's view; the eye spans a wider
    // field, so shrink it by the ratio of the two vertical extents to keep its angular size.
    float scale = 1.0f;
    if (ws.gameTanHalfY > 0.0f && mVrEyeTanHalfY > 0.0f) {
        scale = ws.gameTanHalfY / mVrEyeTanHalfY;
        scale = scale < 0.2f ? 0.2f : (scale > 2.0f ? 2.0f : scale);
    }
    const int32_t left = (int32_t)lroundf((sx - ws.pivotX * scale) * 4.0f);
    const int32_t top = (int32_t)lroundf((sy - ws.pivotY * scale) * 4.0f);
    *ulx = left;
    *uly = top;
    *lrx = left + (int32_t)lroundf(ws.width * scale * 4.0f) - 1;
    *lry = top + (int32_t)lroundf(ws.height * scale * 4.0f) - 1;
    *uls = 0;
    *ult = 0;
    *dsdx = (int16_t)lroundf(1024.0f / scale);
    *dtdy = (int16_t)lroundf(1024.0f / scale);
    float depth = (cz / cw) * 16352.0f + 16352.0f;
    depth = depth < 0.0f ? 0.0f : (depth > 32767.0f ? 32767.0f : depth);
    const uint16_t primDepth = (uint16_t)depth;
    if (primDepth != mRdp->prim_depth) {
        Flush();
        mRdp->prim_depth = primDepth;
    }
    return true;
}

void Interpreter::SetVrEyeTarget(uint32_t width, uint32_t height) {
    if (!mVrEyeActive) {
        mVrSavedDimensions = mCurDimensions;
        mVrSavedMsaaLevel = mMsaaLevel;
        mVrEyeActive = true;
    }
    mCurDimensions.width = width;
    mCurDimensions.height = height;
    mCurDimensions.aspect_ratio = (float)width / (float)height;
    mMsaaLevel = 1;
}

void Interpreter::ClearVrEyeTarget() {
    if (!mVrEyeActive) {
        return;
    }
    mVrEyeActive = false;
    mCurDimensions = mVrSavedDimensions;
    mMsaaLevel = mVrSavedMsaaLevel;
}

// Host hook run after the frame is fully composited into the window back buffer and before the
// swap. Null unless the host installs one (G-Diffuser's OpenXR presenter copies the back buffer
// into a headset swapchain here), so the stock path pays one pointer test.
extern "C" {
void (*gfx_pre_present_hook)(void) = nullptr;
}

void Interpreter::EndFrame() {
    mRapi->EndFrame();
    if (gfx_pre_present_hook != nullptr) {
        gfx_pre_present_hook();
    }
    mWapi->SwapBuffersBegin();
    mRapi->FinishRender();
    mWapi->SwapBuffersEnd();
}

void gfx_set_target_ucode(UcodeHandlers ucode) {
    ucode_handler_index = ucode;
}

int Interpreter::GetTargetFps() {
    return mWapi->GetTargetFps();
}

void Interpreter::SetTargetFps(int fps) {
    mWapi->SetTargetFps(fps);
}

void Interpreter::SetMaxFrameLatency(int latency) {
    mWapi->SetMaxFrameLatency(latency);
}

int Interpreter::CreateFrameBuffer(uint32_t width, uint32_t height, uint32_t native_width, uint32_t native_height,
                                   uint8_t resize, bool forceFixedAspect) {
    uint32_t orig_width = width, orig_height = height;
    if (resize) {
        AdjustWidthHeightForScale(width, height, native_width, native_height);
    }

    int fb = mRapi->CreateFramebuffer();
    mRapi->UpdateFramebufferParameters(fb, width, height, 1, true, true, true, true);

    mFrameBuffers[fb] = {
        orig_width, orig_height, width, height, native_width, native_height, static_cast<bool>(resize), forceFixedAspect
    };
    return fb;
}

void Interpreter::SetFrameBuffer(int fb, float noiseScale) {
    mRapi->StartDrawToFramebuffer(fb, noiseScale);
    mRapi->ClearFramebuffer(false, true);
}

void Interpreter::CopyFrameBuffer(int fb_dst_id, int fb_src_id, bool copyOnce, bool* hasCopiedPtr) {
    // Do not copy again if we have already copied before
    if (copyOnce && hasCopiedPtr != nullptr && *hasCopiedPtr) {
        return;
    }

    if (fb_src_id == 0 && mRendersToFb) {
        // read from the framebuffer we've been rendering to
        fb_src_id = mGameFb;
    }

    int srcX0, srcY0, srcX1, srcY1;
    int dstX0, dstY0, dstX1, dstY1;

    // When rendering to the main window buffer or MSAA is enabled with a buffer size equal to the view port,
    // then the source coordinates must account for any docked ImGui elements
    if (fb_src_id == 0 || (mMsaaLevel > 1 && mCurDimensions.width == mGameWindowViewport.width &&
                           mCurDimensions.height == mGameWindowViewport.height)) {
        srcX0 = mGameWindowViewport.x;
        srcY0 = mGameWindowViewport.y;
        srcX1 = mGameWindowViewport.x + mGameWindowViewport.width;
        srcY1 = mGameWindowViewport.y + mGameWindowViewport.height;
    } else {
        srcX0 = 0;
        srcY0 = 0;
        srcX1 = mCurDimensions.width;
        srcY1 = mCurDimensions.height;
    }

    dstX0 = 0;
    dstY0 = 0;
    dstX1 = mCurDimensions.width;
    dstY1 = mCurDimensions.height;

    mRapi->CopyFramebuffer(fb_dst_id, fb_src_id, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1);

    // Set the copied pointer if we have one
    if (hasCopiedPtr != nullptr) {
        *hasCopiedPtr = true;
    }
}

void Interpreter::ResetFrameBuffer() {
    mRapi->StartDrawToFramebuffer(0, (float)mCurDimensions.height / mNativeDimensions.height);
}

void Interpreter::AdjustPixelDepthCoordinates(float& x, float& y) {
    x = x * RATIO_X(mActiveFrameBuffer, mCurDimensions) -
        (mNativeDimensions.width * RATIO_X(mActiveFrameBuffer, mCurDimensions) - mCurDimensions.width) / 2;
    y *= RATIO_Y(mActiveFrameBuffer, mCurDimensions);
    if (!mRendersToFb || (mMsaaLevel > 1 && mCurDimensions.width == mGameWindowViewport.width &&
                          mCurDimensions.height == mGameWindowViewport.height)) {
        x += mGameWindowViewport.x;
        y += mGfxCurrentWindowDimensions.height - (mGameWindowViewport.y + mGameWindowViewport.height);
    }
}

void Interpreter::GetPixelDepthPrepare(float x, float y) {
    AdjustPixelDepthCoordinates(x, y);
    mGetPixelDepthPending.emplace(x, y);
}

uint16_t Interpreter::GetPixelDepth(float x, float y) {
    AdjustPixelDepthCoordinates(x, y);

    if (auto it = mGetPixelDepthCached.find(std::make_pair(x, y)); it != mGetPixelDepthCached.end()) {
        return it->second;
    }

    mGetPixelDepthPending.emplace(x, y);

    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> res =
        mRapi->GetPixelDepth(mRendersToFb ? mGameFb : 0, mGetPixelDepthPending);
    mGetPixelDepthCached.merge(res);
    mGetPixelDepthPending.clear();

    return mGetPixelDepthCached.find(std::make_pair(x, y))->second;
}

void gfx_push_current_dir(char* path) {
    if (gfx_check_image_signature(path) == 1)
        path = &path[7];

    currentDir.push(GetPathWithoutFileName(path));
}

int32_t gfx_check_image_signature(const char* imgData) {
    uintptr_t i = (uintptr_t)(imgData);

    if ((i & 1) == 1) {
        return 0;
    }

    // Filter addresses that are obviously not valid string pointers before
    // attempting to dereference for the "__OTR__" check.
    if (i == 0 || i < 0x10000) {
        return 0;
    }
#if UINTPTR_MAX > 0xFFFFFFFFu
    // On 64-bit: filter kernel/sentinel addresses. Upper bound covers all
    // user-space layouts (x86_64 47-bit canonical, ARM64 48-bit VA, etc.).
    if (i > 0x0000FFFFFFFFFFFFull) {
        return 0;
    }
#endif

#ifdef _WIN32
    // Plausible-looking but unmapped pointers still reach here (stale SETTIMG
    // addresses during screen transitions crashed OtrSignatureCheck's strncmp).
    // Verify the memory is actually committed and readable first.
    {
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery(reinterpret_cast<const void*>(imgData), &mbi, sizeof(mbi)) == 0 ||
            mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) {
            return 0;
        }
    }
#endif

    return Ship::Context::GetInstance()->GetResourceManager()->OtrSignatureCheck(imgData);
}

void Interpreter::RegisterBlendedTexture(const char* name, uint8_t* mask, uint8_t* replacement) {
    if (gfx_check_image_signature(name)) {
        name += 7;
    }

    if (gfx_check_image_signature(reinterpret_cast<char*>(replacement))) {
        Fast::Texture* tex = std::static_pointer_cast<Fast::Texture>(
                                 Ship::Context::GetInstance()->GetResourceManager()->LoadResourceProcess(
                                     reinterpret_cast<char*>(replacement)))
                                 .get();

        replacement = tex->ImageData;
    }

    mMaskedTextures[name] = MaskedTextureEntry{ mask, replacement };
}

void Interpreter::UnregisterBlendedTexture(const char* name) {
    if (gfx_check_image_signature(name)) {
        name += 7;
    }

    mMaskedTextures.erase(name);
}

// New getters and setters
void Interpreter::SetNativeDimensions(float width, float height) {
    mNativeDimensions.width = width;
    mNativeDimensions.height = height;
}

void Interpreter::SetResolutionMultiplier(float multiplier) {
    mCurDimensions.internal_mul = multiplier;
}

void Interpreter::SetMsaaLevel(uint32_t level) {
    mMsaaLevel = level;
}

void Interpreter::GetCurDimensions(uint32_t* width, uint32_t* height) {
    *width = mCurDimensions.width;
    *height = mCurDimensions.height;
}

} // namespace Fast

void gfx_cc_get_features(uint64_t shader_id0, uint64_t shader_id1, struct CCFeatures* cc_features) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 4; k++) {
                cc_features->c[i][j][k] = shader_id0 >> i * 32 + j * 16 + k * 4 & 0xf;
            }
        }
    }

    cc_features->opt_alpha = (shader_id1 & SHADER_OPT(ALPHA)) != 0;
    cc_features->opt_fog = (shader_id1 & SHADER_OPT(FOG)) != 0;
    cc_features->opt_texture_edge = (shader_id1 & SHADER_OPT(TEXTURE_EDGE)) != 0;
    cc_features->opt_noise = (shader_id1 & SHADER_OPT(NOISE)) != 0;
    cc_features->opt_2cyc = (shader_id1 & SHADER_OPT(_2CYC)) != 0;
    cc_features->opt_alpha_threshold = (shader_id1 & SHADER_OPT(ALPHA_THRESHOLD)) != 0;
    cc_features->opt_invisible = (shader_id1 & SHADER_OPT(INVISIBLE)) != 0;
    cc_features->opt_grayscale = (shader_id1 & SHADER_OPT(GRAYSCALE)) != 0;
    cc_features->opt_prim_depth = (shader_id1 & SHADER_OPT(PRIM_DEPTH)) != 0;

    cc_features->clamp[0][0] = shader_id1 & SHADER_OPT(TEXEL0_CLAMP_S);
    cc_features->clamp[0][1] = shader_id1 & SHADER_OPT(TEXEL0_CLAMP_T);
    cc_features->clamp[1][0] = shader_id1 & SHADER_OPT(TEXEL1_CLAMP_S);
    cc_features->clamp[1][1] = shader_id1 & SHADER_OPT(TEXEL1_CLAMP_T);

    cc_features->usedTextures[0] = false;
    cc_features->usedTextures[1] = false;
    cc_features->used_masks[0] = false;
    cc_features->used_masks[1] = false;
    cc_features->used_blend[0] = false;
    cc_features->used_blend[1] = false;
    cc_features->numInputs = 0;

    for (int c = 0; c < 2; c++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 4; j++) {
                if (cc_features->c[c][i][j] >= SHADER_INPUT_1 && cc_features->c[c][i][j] <= SHADER_INPUT_7) {
                    if (cc_features->c[c][i][j] > cc_features->numInputs) {
                        cc_features->numInputs = cc_features->c[c][i][j];
                    }
                }
                if (cc_features->c[c][i][j] == SHADER_TEXEL0 || cc_features->c[c][i][j] == SHADER_TEXEL0A) {
                    cc_features->usedTextures[0] = true;
                    if (cc_features->opt_2cyc) {
                        cc_features->usedTextures[1] = true;
                    }
                }
                if (cc_features->c[c][i][j] == SHADER_TEXEL1 || cc_features->c[c][i][j] == SHADER_TEXEL1A) {
                    cc_features->usedTextures[1] = true;
                    if (cc_features->opt_2cyc) {
                        cc_features->usedTextures[0] = true;
                    }
                }
            }
        }
    }

    for (int c = 0; c < 2; c++) {
        cc_features->do_single[c][0] = cc_features->c[c][0][2] == SHADER_0;
        cc_features->do_single[c][1] = cc_features->c[c][1][2] == SHADER_0;
        cc_features->do_multiply[c][0] = cc_features->c[c][0][1] == SHADER_0 && cc_features->c[c][0][3] == SHADER_0;
        cc_features->do_multiply[c][1] = cc_features->c[c][1][1] == SHADER_0 && cc_features->c[c][1][3] == SHADER_0;
        cc_features->do_mix[c][0] = cc_features->c[c][0][1] == cc_features->c[c][0][3];
        cc_features->do_mix[c][1] = cc_features->c[c][1][1] == cc_features->c[c][1][3];
        cc_features->color_alpha_same[c] = (shader_id0 >> c * 32 & 0xffff) == (shader_id0 >> c * 32 + 16 & 0xffff);
    }

    if (cc_features->usedTextures[0] && shader_id1 & SHADER_OPT(TEXEL0_MASK)) {
        cc_features->used_masks[0] = true;
    }
    if (cc_features->usedTextures[1] && shader_id1 & SHADER_OPT(TEXEL1_MASK)) {
        cc_features->used_masks[1] = true;
    }

    if (cc_features->usedTextures[0] && shader_id1 & SHADER_OPT(TEXEL0_BLEND)) {
        cc_features->used_blend[0] = true;
    }
    if (cc_features->usedTextures[1] && shader_id1 & SHADER_OPT(TEXEL1_BLEND)) {
        cc_features->used_blend[1] = true;
    }

    cc_features->shader_id = Fast::ShaderIdUnmask(shader_id1);
}

extern "C" int gfx_create_framebuffer(uint32_t width, uint32_t height, uint32_t native_width, uint32_t native_height,
                                      uint8_t resize, bool forceFixedAspect) {
    return Fast::mInstance.lock().get()->CreateFrameBuffer(width, height, native_width, native_height, resize,
                                                           forceFixedAspect);
}

extern "C" void gfx_texture_cache_clear() {
    Fast::mInstance.lock().get()->TextureCacheClear();
}

extern "C" void gfx_shader_cache_clear() {
    auto instance = Fast::mInstance.lock().get();
    instance->mColorCombinerPool.clear();
    instance->mPrevCombiner = Fast::mInstance.lock().get()->mColorCombinerPool.end();
    instance->mRenderingState.mShaderProgram = nullptr;
    instance->mRapi->ClearShaderCache();
}

extern "C" void gfx_register_fb_texture(const void* cpuAddr, int fbId) {
    Fast::mInstance.lock().get()->RegisterFbTexture(cpuAddr, fbId);
}

extern "C" void gfx_unregister_fb_texture(const void* cpuAddr) {
    Fast::mInstance.lock().get()->UnregisterFbTexture(cpuAddr);
}
