#version 430
#define USE_VOXEL_OBJECT_INSTANCING
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_gpu_shader_int64 : require

struct Object {
    mat4 m_model;
    mat4 m_model_inverse;
    mat4 m_prev_model;
    vec4 m_dimensions;
};

layout(std430, binding = 0) buffer Instances {
    Object objects[];
};

struct TextureInformation {
    uint64_t voxel_texture_handle;
    uint64_t palette_handle;
    uint64_t material_handle;
};

layout(std430, binding = 1) buffer Textures {
    TextureInformation texture_infos[];
};

#if defined VERTEX_SHADER

in vec3 in_position;

uniform mat4 m_camera;
uniform mat4 m_proj;

#if USE_VOXEL_OBJECT_INSTANCING == 0
uniform int u_instanceID;
#else
int u_instanceID = gl_InstanceID;
#endif

flat out int v_instanceID;

void main() {
    mat4 m_model = objects[u_instanceID].m_model;
    mat4 m_view = m_camera * m_model;
    vec4 p = m_view * (vec4(in_position, 1.0) * objects[u_instanceID].m_dimensions);
    gl_Position = m_proj * p;
    v_instanceID = u_instanceID;
}

#elif defined FRAGMENT_SHADER

#include programs/pcg_random.glsl
#include programs/utils.glsl
#line 22

flat in int v_instanceID;
#if USE_VOXEL_OBJECT_INSTANCING == 0
layout(binding = 0) uniform usampler3D u_voxel_data;
layout(binding = 1) uniform sampler2D u_palette_data;
layout(binding = 2) uniform sampler2D u_material_data;
#endif
layout(binding = 3) uniform sampler2D u_prev_linear_depth;
uniform mat4 uInvView;
uniform mat4 uInvProjection;
uniform mat4 m_camera;
uniform mat4 m_proj;
uniform mat4 m_prev_viewproj;
uniform int frame_counter;

layout(location = 0) out vec3 u_albedo;
layout(location = 1) out vec3 u_normal;
layout(location = 2) out float u_linear_depth;
layout(location = 3) out vec4 u_material;
layout(location = 4) out vec2 u_motion_vector;

float worldPosToDepth(vec3 worldPos) {
    mat4 viewProj = m_proj * m_camera;
    vec4 clipPos = viewProj * vec4(worldPos, 1.0);
    float ndcDepth = clipPos.z / clipPos.w;
    // OpenGL NDC z [-1, 1] -> depth buffer [0, 1]
    return ndcDepth * 0.5 + 0.5;
}

vec3 encodeNormalRGB10A2(vec3 normal) {
    return normal * 0.5 + 0.5;
}

vec2 compute_motion_vector(
    vec2 screen_uv,
    vec3 localPos,
    mat4 prevModel,
    mat4 prevViewProj
) {
    vec4 prevWorldPos = prevModel * vec4(localPos, 1.0);
    vec4 prevClip = prevViewProj * prevWorldPos;
    vec2 prevNDC = prevClip.xy / prevClip.w;
    vec2 prevUV = prevNDC * 0.5 + 0.5;
    return prevUV - screen_uv;
}

void main() {
    Object object = objects[v_instanceID];
    mat4 m_model = object.m_model;
    mat4 m_model_inverse = object.m_model_inverse;
    mat4 m_prev_model = object.m_prev_model;

    #if USE_VOXEL_OBJECT_INSTANCING == 1
    TextureInformation texture_info = texture_infos[v_instanceID];
    usampler3D u_voxel_data = usampler3D(texture_info.voxel_texture_handle);
    sampler2D u_palette_data = sampler2D(texture_info.palette_handle);
    sampler2D u_material_data = sampler2D(texture_info.material_handle);
    #endif

    float inv_palette_size = 1.0 / (textureSize(u_palette_data, 0).r);
    Box bbox = Box(vec3(0.0), vec3(textureSize(u_voxel_data, 0)));
    vec3 size = bbox.max - bbox.min;
    int MAX_STEPS = int(max(size.x, max(size.y, size.z))) * 3;

    vec2 screen_uv = gl_FragCoord.xy / SCREEN_DIMENSIONS;
    // TODO(david): Deactivated pixel jittering for now, reactivate once final image TAA is implemented
    Ray camera_ray = compute_camera_ray(screen_uv, uInvProjection, uInvView, frame_counter, 0.0);
    Ray local_ray = transform_to_local_ray(camera_ray, m_model_inverse);

    float t;
    if (intersectAABB(local_ray, bbox, t)) {
        float zbuffer_depth = texture(u_prev_linear_depth, screen_uv).r;
        if (t >= zbuffer_depth) {
            discard;
        }
        vec3 bbox_hit = local_ray.origin + (t - 0.01) * local_ray.direction;
        Ray bbox_ray = Ray(bbox_hit, local_ray.direction);
        Hit hit = dda(bbox_ray, MAX_STEPS, u_voxel_data, bbox);
        if (hit.hit) {
            ivec2 palette_coord = ivec2(voxelmap(hit.voxel, bbox, u_voxel_data), 0);
            vec3 world_space_hit = (m_model * vec4(hit.position, 1.0)).xyz;
            vec4 material = texelFetch(u_material_data, palette_coord, 0).rgba;
            if (material.a < 0.0) {
                // TODO(david): glass rendering, maybe skip voxels in dda
                discard;
            }
            u_albedo = texelFetch(u_palette_data, palette_coord, 0).rgb;
            u_normal = normalize((m_model * vec4(hit.normal, 0.0)).xyz);
            u_linear_depth = distance(local_ray.origin, hit.position);
            u_material = material;
            u_motion_vector = compute_motion_vector(screen_uv, hit.position, m_prev_model, m_prev_viewproj);
            gl_FragDepth = worldPosToDepth(world_space_hit);
        } else {
            discard;
        }
    } else {
        discard;
    }
}

#endif
