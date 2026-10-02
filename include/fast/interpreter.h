#pragma once

#include <stdbool.h>
#include <stdint.h>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <list>
#include <limits>
#include <cstddef>
#include <vector>
#include <stack>
#include <string>
#include <string_view>
#include <memory>

#include "fast/lus_gbi.h"
#include "fast/types.h"
#include "fast/ucodehandlers.h"
#include "backends/gfx_rendering_api.h"
#include "fast/debug/GfxDebugger.h"

#include "fast/resource/type/Texture.h"
#include "ship/resource/Resource.h"

// TODO figure out why changing these to 640x480 makes the game only render in a quarter of the window
#define SCREEN_WIDTH 320
#define SCREEN_HEIGHT 240

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <compare>
#endif

/*enum {
    CC_0,
    CC_TEXEL0,
    CC_TEXEL1,
    CC_PRIM,
    CC_SHADE,
    CC_ENV,
    CC_TEXEL0A,
    CC_LOD
};*/

enum {
    SHADER_0,
    SHADER_INPUT_1,
    SHADER_INPUT_2,
    SHADER_INPUT_3,
    SHADER_INPUT_4,
    SHADER_INPUT_5,
    SHADER_INPUT_6,
    SHADER_INPUT_7,
    SHADER_TEXEL0,
    SHADER_TEXEL0A,
    SHADER_TEXEL1,
    SHADER_TEXEL1A,
    SHADER_1,
    SHADER_COMBINED,
    SHADER_NOISE
};

#ifdef __cplusplus
enum class ShaderOpts {
    ALPHA,
    FOG,
    TEXTURE_EDGE,
    NOISE,
    _2CYC,
    ALPHA_THRESHOLD,
    INVISIBLE,
    GRAYSCALE,
    TEXEL0_CLAMP_S,
    TEXEL0_CLAMP_T,
    TEXEL1_CLAMP_S,
    TEXEL1_CLAMP_T,
    TEXEL0_MASK,
    TEXEL1_MASK,
    TEXEL0_BLEND,
    TEXEL1_BLEND,
    PRIM_DEPTH,
    PRISM_SHADER, // 16-bit width
    MAX
};

#define SHADER_OPT(opt) ((uint64_t)(1 << static_cast<int>(ShaderOpts::opt)))
#endif

struct ColorCombinerKey {
    uint64_t combine_mode;
    uint64_t options;
    uint64_t shader_id;

#ifdef __cplusplus
    auto operator<=>(const ColorCombinerKey&) const = default;
#endif
};

#define SHADER_MAX_TEXTURES 6
#define SHADER_FIRST_TEXTURE 0
#define SHADER_FIRST_MASK_TEXTURE 2
#define SHADER_FIRST_REPLACEMENT_TEXTURE 4

struct CCFeatures {
    int c[2][2][4];
    bool opt_alpha;
    bool opt_fog;
    bool opt_texture_edge;
    bool opt_noise;
    bool opt_2cyc;
    bool opt_alpha_threshold;
    bool opt_invisible;
    bool opt_grayscale;
    bool opt_prim_depth;
    bool usedTextures[2];
    bool used_masks[2];
    bool used_blend[2];
    bool clamp[2][2];
    int numInputs;
    bool do_single[2][2];
    bool do_multiply[2][2];
    bool do_mix[2][2];
    bool color_alpha_same[2];
    int16_t shader_id;
};

void gfx_cc_get_features(uint64_t shader_id0, uint64_t shader_id1, struct CCFeatures* cc_features);

union Gfx;

namespace Fast {

class GfxRenderingAPI;
class GfxWindowBackend;

constexpr size_t MAX_SEGMENT_POINTERS = 16;
constexpr size_t SHADER_ID_SHIFT = 17;
constexpr int16_t ShaderIdUnmask(int id) {
    return (id >> SHADER_ID_SHIFT) & 0xFFFF;
}

struct GfxExecStack {
    // This is a dlist stack used to handle dlist calls.
    std::stack<F3DGfx*> cmd_stack = {};
    // This is also a dlist stack but a std::vector is used to make it possible
    // to iterate on the elements.
    // The purpose of this is to identify an instruction at a poin in time
    // which would not be possible with just a F3DGfx* because a dlist can be called multiple times
    // what we do instead is store the call path that leads to the instruction (including branches)
    std::vector<const F3DGfx*> gfx_path = {};
    struct CodeDisp {
        const char* file;
        int line;
    };
    // stack for OpenDisp/CloseDisps
    std::vector<CodeDisp> disp_stack{};

    void start(F3DGfx* dlist);
    void stop();
    F3DGfx*& currCmd();
    void openDisp(const char* file, int line);
    void closeDisp();
    const std::vector<CodeDisp>& getDisp() const;
    void branch(F3DGfx* caller);
    void call(F3DGfx* caller, F3DGfx* callee);
    F3DGfx* ret();
};

struct XYWidthHeight {
    int16_t x, y;
    uint32_t width, height;
};

struct GfxDimensions {
    float internal_mul;
    uint32_t width, height;
    float aspect_ratio;
};

struct TextureCacheKey {
    const uint8_t* texture_addr;
    const uint8_t* palette_addrs[2];
    uint8_t fmt, siz;
    uint8_t palette_index;
    uint32_t size_bytes;
    uint32_t line_size_bytes;
    uint32_t full_image_line_size_bytes;
    uint16_t tile_width;
    uint16_t tile_height;
    uint8_t cms;
    uint8_t cmt;
    uint8_t masks;
    uint8_t maskt;
    bool force_opaque_alpha = false;
    // The same source address can hold different TMEM contents across a frame's load ordering,
    // so an address-only key serves a stale decode.
    uint32_t tmem_content_hash = 0;

    // palette_addrs distinguishes the same texture drawn with different palettes, but a menu
    // fade rewrites the palette content in place at one address, so an address-only key freezes
    // the fade. Left 0 unless GDX_CI_PALETTE_HASH is set, pending validation.
    uint32_t palette_content_hash = 0;
    uint8_t resource_bit_offset = 0;

    bool operator==(const TextureCacheKey&) const noexcept = default;

    struct Hasher {
        size_t operator()(const TextureCacheKey& key) const noexcept {
            uintptr_t addr = (uintptr_t)key.texture_addr;
            return (size_t)(addr ^ (addr >> 5));
        }
    };
};

typedef std::unordered_map<TextureCacheKey, struct TextureCacheValue, TextureCacheKey::Hasher> TextureCacheMap;
typedef std::pair<const TextureCacheKey, struct TextureCacheValue> TextureCacheNode;

struct TextureCacheValue {
    uint32_t texture_id;
    uint8_t cms, cmt;
    bool linear_filter;
    uint64_t rgba16_opaque_pixels = 0;
    uint64_t rgba16_transparent_pixels = 0;
    uint64_t rgba16_forced_opaque_pixels = 0;

    // A TextureCacheLookup insert only reserves the slot; UploadTexture creates the GPU
    // resource, and some decode paths bail out before reaching it (a transiently-null CI palette
    // slot, a zero-sized load). Lookup treats an unuploaded hit as a miss and retries the decode
    // rather than serving a texture that was never created.
    bool uploaded = false;

    std::list<struct TextureCacheMapIter>::iterator lru_location;
};

struct TextureCacheMapIter {
    TextureCacheMap::iterator it;
};

struct RGBA {
    uint8_t r, g, b, a;
};

struct LoadedVertex {
    float x, y, z, w;
    float u, v;
    struct RGBA color;
    uint8_t clip_rej;
};

struct RawTexMetadata {
    uint16_t width, height;
    float h_byte_scale = 1, v_pixel_scale = 1;
    std::shared_ptr<Fast::Texture> resource;
    Fast::TextureType type;
};

struct LoadedTexture {
    const uint8_t* addr;
    uint32_t orig_size_bytes;
    uint32_t size_bytes;
    uint32_t full_image_line_size_bytes;
    uint32_t line_size_bytes;
    uint32_t tex_flags;
    struct RawTexMetadata raw_tex_metadata;
    bool masked;
    bool blended;
    uint16_t tmem_start;
    uint16_t tmem_word_count;
    // Resource views retain native texel units and TMEM padding, unlike physical source rows.
    uint32_t resource_width;
    uint32_t resource_height;
    uint32_t resource_tmem_line_bytes;
    uint8_t resource_bit_offset;
};

#define MAX_LIGHTS 32
#define MAX_VERTICES 64

struct RSP {
    float modelview_matrix_stack[11][4][4];
    uint8_t modelview_matrix_stack_size;

    float MP_matrix[4][4];
    float P_matrix[4][4];

    F3DLight_t lookat[2];
    F3DLight current_lights[MAX_LIGHTS + 1];
    float current_lights_coeffs[MAX_LIGHTS][3];
    float current_lookat_coeffs[2][3]; // lookat_x, lookat_y
    uint8_t current_num_lights;        // includes ambient light
    bool lights_changed;

    uint32_t geometry_mode;
    int16_t fog_mul, fog_offset;

    uint32_t extra_geometry_mode;
    uintptr_t branch_z_target;
    float viewport_z_scale;
    float viewport_z_trans;

    struct {
        // U0.16
        uint16_t s, t;
    } texture_scaling_factor;

    struct LoadedVertex loaded_vertices[MAX_VERTICES + 4];

    uint8_t dmem[4096];
    uint16_t dma_io_dmem;
    // DMEM does not persist across tasks, so SpReset() clears this per Run().
    bool dma_io_loaded;
    float f3dflx_alpha_light[3];
    bool f3dflx_alpha_light_valid;
};

struct RDP {
    const uint8_t* palettes[2];
    // Original DRAM source address of the most recent TLUT load per palette half.
    // Used in texture cache keys instead of palettes[] (which always points to staging).
    const uint8_t* palette_dram_addr[2];
    // CI4 palette staging buffer: N64 TMEM holds up to 16 CI4 palettes (16 entries x 2 bytes each = 32 bytes per
    // palette). palettes[0] covers indices 0-7 (256 bytes), palettes[1] covers 8-15 (256 bytes). GfxDpLoadTlut copies
    // TLUT data here at the correct offset so multi-palette CI4 models work.
    uint8_t palette_staging[2][256];
    struct {
        const uint8_t* addr;
        uint8_t siz;
        uint32_t width;
        uint32_t tex_flags;
        struct RawTexMetadata raw_tex_metadata;
    } texture_to_load;
    // One entry per 64-bit TMEM word address: multiple render tiles can point at independent
    // loads within the same 4 KiB image.
    LoadedTexture loaded_texture[512];
    // Emulated 4 KiB TMEM. Import decodes from here using only tile-descriptor state, like
    // hardware, which makes slot-reuse-heavy frames immune to stale per-slot load bookkeeping.
    uint8_t tmem[4096];
    // Bumped on every TMEM write so the texture cache can key on content.
    uint32_t tmem_generation;
    struct {
        uint8_t fmt;
        uint8_t siz;
        uint8_t cms, cmt;
        uint8_t masks, maskt;
        uint8_t shifts, shiftt;
        float uls, ult, lrs, lrt;
        uint16_t tmem; // 0-511, in 64-bit word units
        uint32_t line_size_bytes;
        uint8_t palette;
        uint16_t tmem_index; // Exact TMEM base in 64-bit word units (0-511)
    } texture_tile[8];
    bool textures_changed[2];

    uint8_t first_tile_index;

    uint32_t other_mode_l, other_mode_h;
    uint64_t combine_mode;
    bool grayscale;

    uint8_t prim_lod_fraction;
    uint16_t prim_depth;
    struct RGBA env_color, prim_color, fog_color, blend_color, fill_color, grayscale_color;

    // Chroma key parameters (G_SETKEYR / G_SETKEYGB)
    struct RGBA key_center;
    struct RGBA key_scale;
    int16_t convert_k[6]; // YUV convert coefficients (G_SETCONVERT) — K0-K5

    struct XYWidthHeight viewport, scissor;
    bool viewport_or_scissor_changed;
    void* z_buf_address;
    void* color_image_address;
};

typedef enum Attribute {
    MTX_PROJECTION,
    MTX_LOAD,
    MTX_PUSH,
    MTX_NOPUSH,
    CULL_FRONT,
    CULL_BACK,
    CULL_BOTH,
    MV_VIEWPORT,
    MV_LIGHT,
} Attribute;

extern GfxExecStack g_exec_stack;

struct GfxTextureCache {
    TextureCacheMap map;
    std::list<TextureCacheMapIter> lru;
    std::vector<uint32_t> free_texture_ids;
};

struct ColorCombiner {
    uint64_t shader_id0;
    uint64_t shader_id1;
    bool usedTextures[2];
    struct ShaderProgram* prg[16];
    uint8_t shader_input_mapping[2][7];
};

struct RenderingState {
    uint8_t depth_test_and_mask; // 1: depth test, 2: depth mask
    bool decal_mode;
    bool alpha_blend;
    struct XYWidthHeight viewport, scissor;
    struct ShaderProgram* mShaderProgram;
    TextureCacheNode* mTextures[SHADER_MAX_TEXTURES];
    bool sampler_valid[SHADER_MAX_TEXTURES];
    bool sampler_linear_filter[SHADER_MAX_TEXTURES];
    uint8_t sampler_cms[SHADER_MAX_TEXTURES];
    uint8_t sampler_cmt[SHADER_MAX_TEXTURES];
};

struct FBInfo {
    uint32_t orig_width, orig_height;       // Original shape
    uint32_t applied_width, applied_height; // Up-scaled for the viewport
    uint32_t native_width, native_height;   // Max "native" size of the screen, used for up-scaling
    bool resize;                            // Scale to match the viewport
    bool forceFixedAspect;                  // Preserve aspect ratio even if resize is true
};

struct MaskedTextureEntry {
    uint8_t* mask;
    uint8_t* replacementData;
};

struct GeometryDiagnostics {
    uint64_t verticesLoaded = 0;
    uint64_t invalidVertices = 0;
    uint64_t verticesNonPositiveW = 0;
    uint64_t verticesOutsideNear = 0;
    uint64_t verticesOutsideFar = 0;
    float minNdcX = 0.0f;
    float maxNdcX = 0.0f;
    float minNdcY = 0.0f;
    float maxNdcY = 0.0f;
    float minNdcZ = 0.0f;
    float maxNdcZ = 0.0f;
    // Content fingerprints for sub-frame replay comparison. Replaying a tick's display list with
    // the interpolation fraction pinned should be idempotent and is not: pass 0 clip-rejects ~15
    // fewer triangles than every later pass. Triangle counters cannot say why, since clip_rej
    // derives from the transformed position. vertexHash covers every transformed vertex and its
    // clip flags; mpFirstHash captures MP_matrix at the pass's first vertex, separating "already
    // different before the walk" from "drifted during it".
    uint64_t vertexHash = 0;
    uint64_t mpFirstHash = 0;
    uint64_t trianglesSubmitted = 0;
    uint64_t trianglesClipRejected = 0;
    uint64_t trianglesCullRejected = 0;
    uint64_t trianglesInvisible = 0;
    uint64_t trianglesEmitted = 0;
    // Survived Reject-variant screening with a screen extent beyond ~0.9 NDC. The first per
    // frame is captured below.
    uint64_t bigTriangles = 0;
    float bigTriX[3] = {};
    float bigTriY[3] = {};
    float bigTriZ[3] = {};
    float bigTriW[3] = {};
    uint32_t bigTriGeometryMode = 0;
    uint64_t bigTriCombine = 0;
    const uint8_t* bigTriTexture = nullptr;
    uint8_t bigTriTile = 0;
    float bigTriViewportX = 0.0f;
    float bigTriViewportY = 0.0f;
    float bigTriViewportW = 0.0f;
    float bigTriViewportH = 0.0f;
    uint64_t dmaIoLoads = 0;
    uint64_t f3dflxAlphaVertices = 0;
    uint64_t gpuDrawCalls = 0;
    uint64_t gpuTriangles = 0;
    uint64_t variantSwitches = 0;
    uint64_t preFlxVertices = 0;
    uint64_t preFlxTrianglesSubmitted = 0;
    uint64_t preFlxTrianglesEmitted = 0;
    uint64_t preFlxGpuDrawCalls = 0;
    uint64_t preFlxGpuTriangles = 0;
    uint64_t rgba16OpaquePixels = 0;
    uint64_t rgba16TransparentPixels = 0;
    uint64_t rgba16ForcedOpaquePixels = 0;
    uint64_t preFlxRgba16OpaquePixels = 0;
    uint64_t preFlxRgba16TransparentPixels = 0;
    uint64_t preFlxRgba16ForcedOpaquePixels = 0;
    uint64_t depthBypassTriangles = 0;
    uint64_t preFlxDepthBypassTriangles = 0;
    uint64_t texturedTriangles = 0;
    uint64_t texture0BoundTriangles = 0;
    uint64_t texture0MissingTriangles = 0;
    uint64_t forcedSimpleMaterialTriangles = 0;
    uint64_t preFlxTexturedTriangles = 0;
    uint64_t preFlxTexture0BoundTriangles = 0;
    uint64_t preFlxTexture0MissingTriangles = 0;
    uint64_t preFlxForcedSimpleMaterialTriangles = 0;
    uint64_t lastShaderId0 = 0;
    uint64_t lastShaderId1 = 0;
    uint64_t preFlxShaderId0 = 0;
    uint64_t preFlxShaderId1 = 0;
    uint32_t textureWidth = 0;
    uint32_t textureHeight = 0;
    uint32_t textureLineBytes = 0;
    uint32_t textureSizeBytes = 0;
    uint16_t textureTmem = 0;
    uint8_t textureTile = 0;
    uint8_t textureMaskS = 0;
    uint8_t textureMaskT = 0;
    uint16_t textureScaleS = 0;
    uint16_t textureScaleT = 0;
    uint32_t preFlxTextureWidth = 0;
    uint32_t preFlxTextureHeight = 0;
    uint32_t preFlxTextureLineBytes = 0;
    uint32_t preFlxTextureSizeBytes = 0;
    uint16_t preFlxTextureTmem = 0;
    uint8_t preFlxTextureTile = 0;
    uint8_t preFlxTextureMaskS = 0;
    uint8_t preFlxTextureMaskT = 0;
    uint16_t preFlxTextureScaleS = 0;
    uint16_t preFlxTextureScaleT = 0;
    uint64_t fogTriangles = 0;
    uint64_t fogBypassTriangles = 0;
    uint64_t preFlxFogTriangles = 0;
    uint64_t preFlxFogBypassTriangles = 0;
    float minFogFactor = std::numeric_limits<float>::infinity();
    float maxFogFactor = -std::numeric_limits<float>::infinity();
    float preFlxMinFogFactor = std::numeric_limits<float>::infinity();
    float preFlxMaxFogFactor = -std::numeric_limits<float>::infinity();
    int16_t preFlxFogMul = 0;
    int16_t preFlxFogOffset = 0;
    float minTextureU = std::numeric_limits<float>::infinity();
    float maxTextureU = -std::numeric_limits<float>::infinity();
    float minTextureV = std::numeric_limits<float>::infinity();
    float maxTextureV = -std::numeric_limits<float>::infinity();
    float preFlxMinTextureU = std::numeric_limits<float>::infinity();
    float preFlxMaxTextureU = -std::numeric_limits<float>::infinity();
    float preFlxMinTextureV = std::numeric_limits<float>::infinity();
    float preFlxMaxTextureV = -std::numeric_limits<float>::infinity();
    uint32_t preFlxOtherModeH = 0;
    uint32_t preFlxOtherModeL = 0;
    uint64_t preFlxCombineMode = 0;
    const uint8_t* preFlxTexture = nullptr;
};

enum class F3dex2Variant : uint8_t {
    Standard,
    Reject,
    FZeroFlxReject,
};

// Host-side payload for F3DEX2_G_LOAD_UCODE, after the port translates physical N64 microcode
// addresses into a semantic variant switch.
constexpr uintptr_t F3DEX2_VARIANT_SWITCH_MARKER = 0x47445800u;

struct GdxPostShaderPipeline;

class Interpreter {
  public:
    Interpreter();
    ~Interpreter();

    void Init(GfxWindowBackend* wapi, class GfxRenderingAPI* rapi, const char* game_name, bool start_in_fullscreen,
              uint32_t width, uint32_t height, uint32_t posX, uint32_t posY);
    void Destroy();
    void SetGfxDebugger(std::shared_ptr<GfxDebugger> debugger);
    std::shared_ptr<GfxDebugger> GetGfxDebugger() const;
    void GetDimensions(uint32_t* width, uint32_t* height, int32_t* posX, int32_t* posY);
    GfxRenderingAPI* GetCurrentRenderingAPI();
    void StartFrame();
    void RunGuiOnly();
    void Run(Gfx* commands, const std::unordered_map<Mtx*, MtxF>& mtx_replacements);
    void EndFrame();

    // Invoked inside Run() after the per-frame clear and viewport/scissor reset, before the
    // task's commands execute. The port's graphics bridge uses it to seed the boot-logo CPU
    // framebuffer underneath the task's content.
    static void SetPortAfterClearHook(void (*hook)(Interpreter*));
    static void (*sPortAfterClearHook)(Interpreter*);
    void HandleWindowEvents();
    bool IsFrameReady();
    bool ViewportMatchesRendererResolution();
    int GetTargetFps();
    void SetTargetFps(int fps);
    void SetMaxFrameLatency(int latency);
    int CreateFrameBuffer(uint32_t width, uint32_t height, uint32_t native_width, uint32_t native_height,
                          uint8_t resize, bool forceFixedAspect = false);
    void SetFrameBuffer(int fb, float noiseScale);
    void CopyFrameBuffer(int fb_dst_id, int fb_src_id, bool copyOnce, bool* hasCopiedPtr);
    void ResetFrameBuffer();
    void AdjustPixelDepthCoordinates(float& x, float& y);
    void GetPixelDepthPrepare(float x, float y);
    uint16_t GetPixelDepth(float x, float y);
    void RegisterBlendedTexture(const char* name, uint8_t* mask, uint8_t* replacement);
    void UnregisterBlendedTexture(const char* name);

    // Register a CPU address as a mirror of a GPU framebuffer texture.
    // When ImportTexture encounters this address, it uses SelectTextureFb instead
    // of reading from CPU memory — giving full GPU resolution with no readback.
    void RegisterFbTexture(const void* cpuAddr, int fbId);
    void UnregisterFbTexture(const void* cpuAddr);

    void SetNativeDimensions(float width, float height);
    void SetResolutionMultiplier(float multiplier);
    void SetMsaaLevel(uint32_t level);
    void GetCurDimensions(uint32_t* width, uint32_t* height);
    void ResetGeometryDiagnostics();
    const GeometryDiagnostics& GetGeometryDiagnostics() const;
    void SetF3dex2Variant(F3dex2Variant variant);
    // [interp-idem] Deterministic baseline for frame-interpolation replay passes (k > 0).
    // Zeroes every piece of emulated RDP/RSP state that survives Run() and invalidates the
    // renderer-side tracked state so the replay re-applies everything, while deliberately NOT
    // touching the content-keyed GPU texture cache or any live resource handle (the reverted
    // full *mRdp restore did both and broke boost/heal plates and the HUD digit).
    void ResetRdpForReplay();
    // VR eye target. While set, StartFrame/Run render offscreen at exactly width x height with no
    // MSAA, no post shader and no hor+ widescreen correction (the eye projection already defines
    // the field of view), and the finished image is left in the game framebuffer texture
    // (GetGameFramebufferTexture). Clearing restores the desktop dimensions and settings.
    void SetVrEyeTarget(uint32_t width, uint32_t height);
    void ClearVrEyeTarget();
    bool IsVrEyeTargetActive() const {
        return mVrEyeActive;
    }
    // Which part of a display list a VR replay draws, split at the host's HUD marker (a G_NOOP
    // with sub-code kVrHudMarker, emitted right before the race HUD). WorldOnly stops at the
    // marker; HudOnly executes everything for its state but draws nothing until the marker.
    // A list without the marker draws entirely in Full/WorldOnly and not at all in HudOnly.
    enum class VrPass { Full, WorldOnly, HudOnly };
    static constexpr uint32_t kVrHudMarker = 0x56;
    // Screen-space sprites the game draws before the HUD marker are bracketed by the host:
    // kVrHudSpritesBegin/End around sprites that belong to the HUD (drawn by the HudOnly pass even
    // though they precede the marker), kVrWorldSpriteBegin/End around a sprite anchored to a world
    // point (re-placed for each eye in WorldOnly passes from the anchor the resolver returns; the
    // marker's low 16 bits carry the anchor id).
    static constexpr uint32_t kVrHudSpritesBegin = 0x57;
    static constexpr uint32_t kVrHudSpritesEnd = 0x58;
    static constexpr uint32_t kVrWorldSpriteBegin = 0x59;
    static constexpr uint32_t kVrWorldSpriteEnd = 0x5A;
    // World-sprite anchor: world position, pivot offset of the anchor inside the sprite and sprite
    // size, all in native 320x240 pixels.
    struct VrWorldSprite {
        float x, y, z;
        float pivotX, pivotY;
        float width, height;
        float gameTanHalfY; // the game camera's vertical half-FOV tangent the sprite was sized for (0: unknown)
    };
    // Vertical half-FOV tangent of the eye being rendered; world sprites are scaled by
    // gameTanHalfY / this so they keep the angular size they have on a flat screen.
    void SetVrEyeTanHalfY(float tanHalfY) {
        mVrEyeTanHalfY = tanHalfY;
    }
    static void SetVrWorldSpriteResolver(bool (*resolver)(uint32_t id, VrWorldSprite* out));
    void SetVrPass(VrPass pass) {
        mVrPass = pass;
    }
    bool VrHudMarkerSeen() const {
        return mVrMarkerSeen;
    }
    void OnVrHudMarker();
    void OnVrSpriteMarker(uint32_t code, uint32_t id);
    // GL texture name (or backend equivalent) of the last Run()'s offscreen game image, stored
    // top-down; 0 when the frame rendered straight to the window.
    uintptr_t GetGameFramebufferTexture() const {
        return mGfxFrameBuffer;
    }

    // private: TODO make these private
    void Flush();
    ShaderProgram* LookupOrCreateShaderProgram(uint64_t id0, uint64_t id1);
    ColorCombiner* LookupOrCreateColorCombiner(const ColorCombinerKey& key);
    void ShaderCacheClear();
    void TextureCacheClear();
    // Implemented in port/gdx_workshop.cpp; gated inside on gEnhancements.Workshop.TextureDump.
    void GdxDumpDecodedRgba32(int tile, const uint8_t* rgba32, uint32_t width, uint32_t height);
    bool TextureCacheLookup(int i, const TextureCacheKey& key);
    void TextureCacheDelete(const uint8_t* origAddr);
    // Palette-keyed companion to TextureCacheDelete. CI entries are keyed on {index address,
    // palette address}, so rewriting palette content at an unchanged address leaves every decode
    // keyed against it stale. No-op unless the address was seen as a TLUT source.
    void TextureCacheDeletePalette(const uint8_t* paletteAddr);
    void ImportTextureRgba16(int textureUnit, int tile, bool importReplacement, bool forceOpaqueAlpha);
    void LoadScaledTexture(uint8_t tile, uint32_t x, uint32_t y, uint32_t width, uint32_t height, bool block);
    void ImportTextureRgba32(int tile, bool importReplacement);
    void ImportTextureIA4(int tile, bool importReplacement);
    void ImportTextureIA8(int tile, bool importReplacement);
    void ImportTextureIA16(int tile, bool importReplacement);
    void ImportTextureI4(int tile, bool importReplacement);
    void ImportTextureI8(int tile, bool importReplacement);
    void ImportTextureCi4(int tile, bool importReplacement);
    void ImportTextureCi8(int tile, bool importReplacement);
    void ImportTextureRaw(int tile, bool importReplacement);
    void ImportTextureImg(int tile, bool importReplacement);
    void ImportTexture(int i, int tile, bool importReplacement);
    void ImportTextureMask(int i, int tile);
    void CalculateNormalDir(const F3DLight_t*, float coeffs[3]);

    void GfxSpMatrix(uint8_t params, const int32_t* addr);
    void GfxSpPopMatrix(uint32_t count);
    void GfxSpVertex(size_t numVertices, size_t destIndex, const F3DVtx* vertices);
    void GfxSpModifyVertex(uint16_t vtxIdx, uint8_t where, uint32_t val);
    void GfxSpTri1(uint8_t vtx1Idx, uint8_t vtx2Idx, uint8_t vtx3Idx, bool isRect);
    void GfxSpLine3DGdx(uint8_t vtx1Idx, uint8_t vtx2Idx, uint8_t halfWidth);
    void GfxSpGeometryMode(uint32_t clear, uint32_t set);
    void GfxSpExtraGeometryMode(uint32_t clear, uint32_t set);
    void GfxSpMovememF3dex2(uint8_t index, uint8_t offset, const void* data);
    void GfxSpDmaIo(bool write, uint16_t dmem, void* data, size_t size);
    void GfxSpMovememF3d(uint8_t index, uint8_t offset, const void* data);
    void GfxSpMovewordF3dex2(uint8_t index, uint16_t offset, uintptr_t data);
    void GfxSpMovewordF3d(uint8_t index, uint16_t offset, uintptr_t data);
    void GfxSpTexture(uint16_t sc, uint16_t tc, uint8_t level, uint8_t tile, uint8_t on);
    void GfxDpSetScissor(uint32_t mode, uint32_t ulx, uint32_t uly, uint32_t lrx, uint32_t lry);
    void GfxDpSetTextureImage(uint32_t format, uint32_t size, uint32_t width, const char* texPath, uint32_t texFlags,
                              RawTexMetadata rawTexMetdata, const void* addr);
    void GfxDpSetTile(uint8_t fmt, uint32_t siz, uint32_t line, uint32_t tmem, uint8_t tile, uint32_t palette,
                      uint32_t cmt, uint32_t maskt, uint32_t shiftt, uint32_t cms, uint32_t masks, uint32_t shifts);
    void GfxDpSetTileSize(uint8_t tile, uint16_t uls, uint16_t ult, uint16_t lrs, uint16_t lrt);
    void GfxDpLoadTlut(uint8_t tile, uint32_t high_index);
    void GfxDpLoadBlock(uint8_t tile, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t dxt);
    void GfxDpLoadTile(uint8_t tile, uint32_t uls, uint32_t ult, uint32_t lrs, uint32_t lrt);
    void StoreLoadedTexture(uint16_t tmemStart, const LoadedTexture& texture);
    void GfxDpSetCombineMode(uint32_t rgb, uint32_t alpha, uint32_t rgb_cyc2, uint32_t alpha_cyc2);
    void GfxDpSetGrayscaleColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    void GfxDpSetEnvColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    void GfxDpSetPrimColor(uint8_t m, uint8_t r, uint8_t l, uint8_t g, uint8_t b, uint8_t a);
    void GfxDpSetFogColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    void GfxDpSetBlendColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a);
    void GfxDpSetFillColor(uint32_t pickedColor);
    void GfxDrawRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry);
    void GfxDpTextureRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry, uint8_t tile, int16_t uls,
                               int16_t ult, int16_t dsdx, int16_t dtdy, bool flip);
    void GfxDpImageRectangle(int32_t tile, int32_t w, int32_t h, int32_t ulx, int32_t uly, int16_t uls, int16_t ult,
                             int32_t lrx, int32_t lry, int16_t lrs, int16_t lrt);
    void GfxDpFillRectangle(int32_t ulx, int32_t uly, int32_t lrx, int32_t lry);
    void GfxDpSetZImage(void* zBufAddr);
    void GfxDpSetColorImage(uint32_t format, uint32_t size, uint32_t width, void* address);
    void GfxSpSetOtherMode(uint32_t shift, uint32_t num_bits, uint64_t mode);
    void GfxDpSetOtherMode(uint32_t h, uint32_t l);

    void Gfxs2dexBgCopy(F3DuObjBg* bg);
    void Gfxs2dexBg1cyc(F3DuObjBg* bg);
    void Gfxs2dexRecyCopy(F3DuObjSprite* spr);

    void AdjustWidthHeightForScale(uint32_t& width, uint32_t& height, uint32_t nativeWidth,
                                   uint32_t nativeHeight) const;
    float AdjXForAspectRatio(float x) const;
    void AdjustVIewportOrScissor(XYWidthHeight* area);
    void CalcAndSetViewport(const F3DVp_t* viewport);

    void SpReset();
    void* SegAddr(uintptr_t w1);

    static const char* CCMUXtoStr(uint32_t ccmux);
    static const char* ACMUXtoStr(uint32_t acmux);
    static void GenerateCC(ColorCombiner* comb, const ColorCombinerKey& key);
    static std::string_view GetBaseTexturePath(std::string_view path);
    static void NormalizeVector(float v[3]);
    static void TransposedMatrixMul(float res[3], const float a[3], const float b[4][4]);
    static void MatrixMul(float res[4][4], const float a[4][4], const float b[4][4]);

    RSP* mRsp;
    RDP* mRdp;
    RenderingState mRenderingState{};

    GfxTextureCache mTextureCache{};
    // TextureCacheDeletePalette has to scan the whole cache, since the map is bucketed by
    // texture_addr rather than by palette, so this set gates the common case -- a plain
    // texture-buffer refresh, never a palette -- out of that scan. A few tens of entries in
    // practice.
    std::unordered_set<const uint8_t*> mSeenPaletteAddrs;
    std::map<ColorCombinerKey, ColorCombiner> mColorCombinerPool; // color_combiner_pool;
    std::map<ColorCombinerKey, ColorCombiner>::iterator mPrevCombiner = mColorCombinerPool.end();
    uint8_t* mTexUploadBuffer = nullptr;

    GfxDimensions mGfxCurrentWindowDimensions{}; // gfx_current_window_dimensions;
    int32_t mCurWindowPosX{};
    int32_t mCurWindowPosY{};
    GfxDimensions mCurDimensions{};        // gfx_current_dimensions;
    GfxDimensions mPrvDimensions{};        // gfx_prev_dimensions;
    XYWidthHeight mGameWindowViewport{};   // gfx_current_game_window_viewport;
    XYWidthHeight mNativeDimensions{};     // gfx_native_dimensions;
    XYWidthHeight mPrevNativeDimensions{}; // gfx_prev_native_dimensions;
    uintptr_t mGfxFrameBuffer{};

    unsigned int mMsaaLevel = 1;
    bool mVrEyeActive = false;
    VrPass mVrPass = VrPass::Full;
    bool mVrSuppressDraw = false;
    bool mVrStopRequested = false;
    bool mVrMarkerSeen = false;
    bool mVrInHudSprites = false;
    bool mVrWorldSpriteActive = false;
    float mVrEyeTanHalfY = 0.0f;
    VrWorldSprite mVrWorldSprite{};
    static bool (*sVrWorldSpriteResolver)(uint32_t id, VrWorldSprite* out);
    bool VrPlaceWorldSprite(int32_t* ulx, int32_t* uly, int32_t* lrx, int32_t* lry, int16_t* uls, int16_t* ult,
                            int16_t* dsdx, int16_t* dtdy);
    GfxDimensions mVrSavedDimensions{};
    unsigned int mVrSavedMsaaLevel = 1;
    bool mDroppedFrame{};
    float* mBufVbo; // 3 vertices in a triangle and 32 floats per vtx
    size_t mBufVboLen{};
    size_t mBufVboNumTris{};
    GfxWindowBackend* mWapi = nullptr;
    GfxRenderingAPI* mRapi = nullptr;
    std::shared_ptr<GfxDebugger> mGfxDebugger;

    uintptr_t mSegmentPointers[MAX_SEGMENT_POINTERS]{};

    bool mFbActive{};
    bool mRendersToFb{}; // game_renders_to_framebuffer;
    // Latched in StartFrame: AdjXForAspectRatio runs per vertex and GfxDrawRectangle per rect,
    // and a CVarGetInteger string-hash lookup on those paths cost ~2 lookups per vertex.
    bool mWidescreenEnabledCache = true;
    bool mForceFixedAspectCache = false;
    bool mWidescreenUiCache = false;
    // Gates the game-side cull widening (gdx_get_ultrawide_cull_xscale) so 16:9-and-below output
    // stays byte-identical with the CVar off. HudMaxAspect optionally confines ANCHOR-scoped HUD
    // elements to a centred band instead of the true corners; see the StartFrame latch.
    bool mUltrawideCache = false;
    bool mRemoveBordersCache = false;
    float mHudMaxAspectCache = 1000.0f;
    // Latched with the graphics CVars above: gEnhancements.Graphics.CRTShader (0 = off,
    // 1 = scanlines, 2 = CRT), gEnhancements.Graphics.CustomShader (non-empty stem), and
    // gEnhancements.Graphics.PostPipeline (path to a .slangp preset). Any active post-process
    // option forces the offscreen render path so the epilogue can downsample the frame and run
    // the shader chain before DrawGame presents.
    int mPostShaderCache = 0;
    std::string mPostPipelineCache;
    std::unique_ptr<GdxPostShaderPipeline> mPostPipeline;
    // Failed-parse cache: a preset that fails to parse must not be re-parsed every frame
    // (re-read + SPDLOG_ERROR per frame stutters and floods the log). Retry only when the
    // preset file changes on disk or the user picks another preset.
    std::string mPostPipelineFailedPath;
    uint64_t mPostPipelineFailedMtime = 0;
    std::map<int, FBInfo>::iterator mActiveFrameBuffer;
    std::map<int, FBInfo> mFrameBuffers;

    int mGameFb{};             // game_framebuffer;
    int mGameFbMsaaResolved{}; // game_framebuffer_msaa_resolved;

    std::set<std::pair<float, float>> mGetPixelDepthPending; // get_pixel_depth_pending;
    std::unordered_map<std::pair<float, float>, uint16_t, hash_pair_ff> mGetPixelDepthCached; // get_pixel_depth_cached;
    std::map<std::string, MaskedTextureEntry, std::less<>> mMaskedTextures;
    std::unordered_map<uintptr_t, int> mFbTextures; // CPU addr -> GPU FB id

    const std::unordered_map<Mtx*, MtxF>* mCurMtxReplacements;
    bool mMarkerOn; // This was originally a debug feature. Now it seems to control s2dex?
    std::unordered_map<size_t, const char*> mShaders;
    GeometryDiagnostics mGeometryDiagnostics{};
    F3dex2Variant mF3dex2Variant = F3dex2Variant::Standard;

    typedef size_t ShaderId;
    std::stack<ShaderId> mShaderStack;
    size_t mShadersIndex;
    int mInterpolationIndex;
    int mInterpolationIndexTarget;
};

void gfx_set_target_ucode(UcodeHandlers ucode);
void gfx_push_current_dir(char* path);
int32_t gfx_check_image_signature(const char* imgData);
const char* gfx_get_shader(int16_t id);
const char* GfxGetOpcodeName(int8_t opcode);

} // namespace Fast

extern "C" void gfx_texture_cache_clear();
extern "C" void gfx_shader_cache_clear();
extern "C" int gfx_create_framebuffer(uint32_t width, uint32_t height, uint32_t native_width, uint32_t native_height,
                                      uint8_t resize, bool forceFixedAspect = false);
