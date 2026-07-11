#version 400

#if defined VERTEX_SHADER

in vec3 in_position;

uniform mat4 m_model;
uniform mat4 m_camera;
uniform mat4 m_proj;

void main() {
    mat4 m_view = m_camera * m_model;
    vec4 p = m_view * vec4(in_position, 1.0);
    gl_Position = m_proj * p;
}

#elif defined FRAGMENT_SHADER

#include programs/pcg_random.glsl
#include programs/utils.glsl

uniform vec3 uCameraPos;
uniform mat4 uInvView;
uniform mat4 uInvProjection;
uniform usampler3D u_voxel_data;
uniform sampler2D u_palette_data;
uniform mat4 m_model;
uniform mat4 m_model_inverse;
uniform mat4 m_camera;
uniform mat4 m_proj;
uniform int frame_counter;

Box bbox = compute_bbox(u_voxel_data);
vec3 size = bbox.max - bbox.min;
int MAX_STEPS = int(max(size.x, max(size.y, size.z))) * 3;
float inv_palette_size = 1.0 / (textureSize(u_palette_data, 0).r - 1.0);

layout(location = 0) out vec3 u_albedo;
layout(location = 1) out vec3 u_normal;
layout(location = 2) out float u_linear_depth;

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

void main() {
    Ray camera_ray = compute_camera_ray(uInvProjection, uInvView, uCameraPos, frame_counter);
    Ray local_ray = transform_to_local_ray(camera_ray, m_model_inverse);

    float t;
    if (intersectAABB(local_ray, bbox, t)) {
        vec3 bbox_hit = local_ray.origin + (t - 0.01) * local_ray.direction;
        Ray bbox_ray = Ray(bbox_hit, local_ray.direction);
        Hit hit = dda(bbox_ray, MAX_STEPS, u_voxel_data, bbox);
        if (hit.hit) {
            vec2 palette_coord = vec2(float(voxelmap(hit.voxel, bbox, u_voxel_data)) * inv_palette_size);
            u_albedo = texture(u_palette_data, palette_coord).rgb;
            u_normal = encodeNormalRGB10A2(normalize((m_model * vec4(hit.normal, 0.0)).xyz));
            u_linear_depth = distance(local_ray.origin, hit.position);
            vec3 world_space_hit = (m_model * vec4(hit.position, 1.0)).xyz;
            gl_FragDepth = worldPosToDepth(world_space_hit);
        } else {
            discard;
        }
    } else {
        discard;
    }
}

#endif
