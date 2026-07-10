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
uniform mat4 m_camera;
uniform mat4 m_proj;

Box bbox = compute_bbox(u_voxel_data);
vec3 size = bbox.max - bbox.min;
int MAX_STEPS = int(max(size.x, max(size.y, size.z))) * 3;
float inv_palette_size = 1.0 / (textureSize(u_palette_data, 0).r - 1.0);

layout(location = 0) out vec3 u_albedo;
layout(location = 1) out vec3 u_normal;

float worldPosToDepth(vec3 worldPos) {
    mat4 viewProj = m_proj * m_camera;
    vec4 clipPos = viewProj * vec4(worldPos, 1.0);
    float ndcDepth = clipPos.z / clipPos.w;
    // OpenGL NDC z [-1, 1] -> depth buffer [0, 1]
    return ndcDepth * 0.5 + 0.5;
}

vec3 encodeNormalRGB10A2(vec3 normal)
{
    return normal * 0.5 + 0.5;
}

void main() {
    Ray camera_ray = compute_camera_ray(uInvProjection, uInvView, uCameraPos);

    float t;
    if (intersectAABB(camera_ray, bbox, t)) {
        vec3 hitPos = camera_ray.origin + (t - 0.01) * camera_ray.direction;
        Ray ray = Ray(hitPos, camera_ray.direction);
        Hit hit = dda(ray, MAX_STEPS, u_voxel_data, bbox);
        if (hit.hit) {
            vec2 palette_coord = vec2(float(voxelmap(hit.voxel, bbox, u_voxel_data)) * inv_palette_size);
            vec3 albedo = texture(u_palette_data, palette_coord).rgb;
            u_albedo = albedo;
            u_normal = encodeNormalRGB10A2(hit.normal);
            gl_FragDepth = worldPosToDepth(hit.position);
        } else {
            discard;
        }
    } else {
        discard;
    }
}

#endif
