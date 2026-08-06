#version 420

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 uv;

void main() {
    gl_Position = vec4(in_position, 1);
    uv = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER
#include programs/pcg_random.glsl
#include programs/utils.glsl
#line 19 3

in vec2 uv;

uniform mat4x4 u_projection_view;
uniform mat4 uInvView;
uniform mat4 uInvProjection;
uniform int frame_counter;
uniform ivec3 occluder_translation;

layout(binding = 0) uniform sampler2D u_normal;
layout(binding = 1) uniform sampler2D u_depth;
layout(binding = 2) uniform sampler2D u_linear_depth;
layout(binding = 3) uniform sampler2D u_material;
layout(binding = 4) uniform usampler3D u_global_occluder;
layout(binding = 5) uniform sampler2D u_last_composite;
layout(binding = 6) uniform sampler2DArray u_stbn_unitvec3;

layout(location = 0) out vec3 out_specular;
layout(location = 1) out float out_reflectivity;

const int MAX_SPECULAR_SAMPLES = 1;
const int MAX_SPECULAR_DISTANCE = 400;
const float MAX_SCREEN_SPACE_REFLECTION_DISTANCE = 30.0;

uint rnd_seed = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * 4097U + uint(frame_counter);
int normal_rand_state = int(rnd_seed) % 64;
float linear_depth = texture(u_linear_depth, uv).r;
vec3 camera_pos = uInvView[3].xyz;
vec3 size = textureSize(u_global_occluder, 0);
Box bbox = Box(vec3(0.0), vec3(size));

vec3 reflect(vec3 I, vec3 N) {
    return I - 2.0 * dot(N, I) * N;
}

vec3 sample_screen_space(vec3 pos, sampler2D tex) {
    vec2 uv = world_to_uv(pos, u_projection_view);
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        return vec3(0.0);
    }
    float screen_depth = texture(u_linear_depth, uv).r;
    float cam_distance = distance(camera_pos, pos);
    vec3 screen_space_color = texture(tex, uv).rgb;
    float reprojection_error = min(distance(screen_depth, cam_distance), 1.0);
    return mix(screen_space_color, vec3(0.0), reprojection_error);
}

vec3 compute_specular_lighting(vec3 pos, vec3 normal, float roughness) {
    vec3 ray_start = pos + normal * 1.0;

    // Specular Lighting
    vec3 specular = vec3(0.0);
    for (int spec_sample = 0; spec_sample < MAX_SPECULAR_SAMPLES; spec_sample += 1) {
        vec3 reflection_vec = reflect(normalize(pos - camera_pos), normal);
        vec3 random_normal = generate_random_stbn_unitvec3(u_stbn_unitvec3, normal_rand_state) * roughness;
        vec3 reflection_jittered = normalize(reflection_vec + random_normal);
        Ray occ_ray = Ray(ray_start, reflection_jittered);
        occ_ray.origin -= occluder_translation;
        Hit occ_hit = sparse_raymarch(occ_ray, MAX_SPECULAR_DISTANCE, u_global_occluder, bbox, 16);
        if (!occ_hit.hit) {
            // TODO(david): sun should also specular reflect
            specular += skyColor(occ_ray.direction, vec3(-1.0));
        } else {
            vec3 global_hit_position = occ_hit.position + occluder_translation;
            if (distance(pos, global_hit_position) <= MAX_SCREEN_SPACE_REFLECTION_DISTANCE) {
                specular += sample_screen_space(global_hit_position, u_last_composite);
            }
        }
    }
    return specular / MAX_SPECULAR_SAMPLES;
}

float fresnel(float f0, vec3 view_dir, vec3 normal) {
    float cosTheta = clamp(dot(normal, -view_dir), 0.0, 1.0);
    return f0 + (1.0 - f0) * pow(1.0 - cosTheta, 5.0);
}

void main() {
    Ray camera_ray = compute_camera_ray(uv, uInvProjection, uInvView, 0, 0.0);
    float depth = texture(u_depth, uv).r;
    vec4 material = texture(u_material, uv);
    float reflectivity = material.r;
    float roughness = material.g;
    float transparency = max(0.0, -material.a);
    if (depth == 1.0) {
        out_specular = vec3(0.0);
        out_reflectivity = 0.0;
        return;
    }
    vec3 normal = texture(u_normal, uv).rgb;
    float f0 = (transparency > 0.0) ? 0.04 : 0.04; // TODO(david): should different material have different fresnel
    float fresnel_factor = fresnel(f0, camera_ray.direction, normal);
    float smooth_reflectivity = mix(0.1, 1.0, (1.0 - roughness)) * fresnel_factor;
    out_reflectivity = mix(smooth_reflectivity, 1.0, reflectivity);
    if (out_reflectivity <= 0.0) {
        out_specular = vec3(0.0);
        return;
    }

    vec3 pos = camera_ray.origin + camera_ray.direction * linear_depth;
    out_specular = compute_specular_lighting(pos, normal, roughness * 0.1);
}
#endif
