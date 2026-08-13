#version 430
#define USE_VOXEL_OBJECT_INSTANCING
#extension GL_ARB_bindless_texture : require
#extension GL_ARB_gpu_shader_int64 : require

layout(std430, binding = 0) buffer InstanceIDs {
    uint ids[];
};

struct Object {
    mat4 m_model;
    mat4 m_model_inverse;
    mat4 m_prev_model;
    vec4 m_dimensions;
};

layout(std430, binding = 1) buffer Instances {
    Object objects[];
};

struct TextureInformation {
    uint64_t voxel_texture_handle;
    uint palette_row;
    uint material_row;
};

layout(std430, binding = 2) buffer Textures {
    TextureInformation texture_infos[];
};

#if defined VERTEX_SHADER

in vec3 in_position;

uniform mat4 m_camera;
uniform mat4 m_proj;

#if USE_VOXEL_OBJECT_INSTANCING == 0
uniform int u_instanceID;
#else
int u_instanceID = int(ids[gl_InstanceID]);
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
#line 55 3

flat in int v_instanceID;
#if USE_VOXEL_OBJECT_INSTANCING == 0
layout(binding = 0) uniform usampler3D u_voxel_data;
uniform int u_palette_row;
uniform int u_material_row;
#endif
layout(binding = 1) uniform sampler2D u_full_palette_texture;
layout(binding = 2) uniform sampler2D u_full_material_texture;
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

uint voxelmap(vec3 p, Box bbox, usampler3D u_voxel_data)
{
    vec3 local_coord = (p + 0.5) / (bbox.max - bbox.min);
    return textureLod(u_voxel_data, local_coord, 0.0).r;
}

bool check_voxel_map_hit(vec3 p, Box bbox, usampler3D u_voxel_data, uint material_row) {
    uint voxel_material = voxelmap(p, bbox, u_voxel_data);
    if (voxel_material == 0) {
        return false;
    }

    ivec2 material_coord = ivec2(voxel_material, material_row);
    float transparency = -texelFetch(u_full_material_texture, material_coord, 0).a;
    if (transparency > 0) {
        ivec2 screen_pos = ivec2(gl_FragCoord.xy);
        if (((screen_pos.x + screen_pos.y) & 1) == 0) {
            return false;
        }
    }
    return true;
}

Hit dda(Ray ray, int max_steps, usampler3D voxels, Box bbox, uint material_row) {
    vec3 pos = ray.origin;
    vec3 rayDir = ray.direction;
    Hit hit;
    hit.hit = false;

    vec3 map = floor(pos);
    vec3 stepDir = vec3(0);
    vec3 sideDist = vec3(9e9);
    vec3 deltaDist = 1. / abs(rayDir);
    float side = 0.;
    vec3 S = step(0., rayDir);

    stepDir = 2. * S - 1.;
    sideDist = (S - stepDir * fract(pos)) * deltaDist;

    bool has_entered = is_inside_box(map, bbox);
    int i;
    for (i = 0; i < max_steps; i++) {
        vec4 conds = step(sideDist.xxyy, sideDist.yzzx);
        vec3 cases = vec3(0);
        cases.x = conds.x * conds.y;
        cases.y = (1. - cases.x) * conds.z * conds.w;
        cases.z = (1. - cases.x) * (1. - cases.y);
        sideDist += max((2. * cases - 1.) * deltaDist, 0.);
        map += cases * stepDir;
        if (is_inside_box(map, bbox) && !has_entered) {
            has_entered = true;
        }
        if (has_entered && !is_inside_box(map, bbox)) {
            return hit;
        }
        if (has_entered && check_voxel_map_hit(map, bbox, voxels, material_row)) // Did we hit anything? if so, we are done!
        {
            side = cases.y + 2. * cases.z;
            break;
        }
    }
    if (!has_entered || i == max_steps) {
        return hit;
    }
    vec3 normal = vec3(0.0);
    normal[int(side)] = -1. * sign(rayDir[int(side)]); // voxel face debug
    vec3 p = map + .5 - stepDir * .5; // Point on axis plane
    float t = (dot(normal, p - pos)) / dot(normal, rayDir);

    hit.hit = true;
    hit.t = t;
    hit.position = pos + rayDir * t;
    hit.voxel = map;
    hit.normal = normal;
    return hit;
}

void main() {
    Object object = objects[v_instanceID];
    mat4 m_model = object.m_model;
    mat4 m_model_inverse = object.m_model_inverse;
    mat4 m_prev_model = object.m_prev_model;

    #if USE_VOXEL_OBJECT_INSTANCING == 1
    TextureInformation texture_info = texture_infos[v_instanceID];
    usampler3D u_voxel_data = usampler3D(texture_info.voxel_texture_handle);
    uint palette_row = texture_info.palette_row;
    uint material_row = texture_info.material_row;
    #else
    int palette_row = u_palette_row;
    int material_row = u_material_row;
    #endif

    Box bbox = Box(vec3(0.0), vec3(textureSize(u_voxel_data, 0)));
    vec3 size = bbox.max - bbox.min;
    int MAX_STEPS = int(max(size.x, max(size.y, size.z))) * 3;

    vec2 screen_uv = gl_FragCoord.xy / SCREEN_DIMENSIONS;
    vec2 jitter = (halton2D(frame_counter) - vec2(0.5)) * 0.5;
    Ray camera_ray = compute_camera_ray(screen_uv, uInvProjection, uInvView, jitter);
    Ray local_ray = transform_to_local_ray(camera_ray, m_model_inverse);

    float t;
    if (intersectAABB(local_ray, bbox, t)) {
        float zbuffer_depth = texture(u_prev_linear_depth, screen_uv).r;
        if (t >= zbuffer_depth) {
            discard;
        }
        vec3 bbox_hit = local_ray.origin + (t - 0.01) * local_ray.direction;
        Ray bbox_ray = Ray(bbox_hit, local_ray.direction);
        Hit hit = dda(bbox_ray, MAX_STEPS, u_voxel_data, bbox, material_row);
        if (hit.hit) {
            uint voxel_material = voxelmap(hit.voxel, bbox, u_voxel_data);

            ivec2 material_coord = ivec2(voxel_material, material_row);
            vec4 material = texelFetch(u_full_material_texture, material_coord, 0).rgba;
            u_material = material;

            ivec2 palette_coord = ivec2(voxel_material, palette_row);
            u_albedo = texelFetch(u_full_palette_texture, palette_coord, 0).rgb;

            vec3 world_space_hit = (m_model * vec4(hit.position, 1.0)).xyz;
            gl_FragDepth = worldPosToDepth(world_space_hit);

            u_normal = normalize((m_model * vec4(hit.normal, 0.0)).xyz);
            u_linear_depth = distance(local_ray.origin, hit.position);
            u_motion_vector = compute_motion_vector(screen_uv, hit.position, m_prev_model, m_prev_viewproj);
            u_motion_vector -= jitter / SCREEN_DIMENSIONS;
        } else {
            discard;
        }
    } else {
        discard;
    }
}

#endif
