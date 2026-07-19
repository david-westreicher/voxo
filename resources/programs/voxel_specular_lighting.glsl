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
#line 19

in vec2 uv;

uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uInvView;
uniform mat4 uInvProjection;
uniform vec3 sun_direction;
uniform int frame_counter;

layout(binding = 0) uniform sampler2D u_normal;
layout(binding = 1) uniform sampler2D u_depth;
layout(binding = 2) uniform sampler2D u_linear_depth;
layout(binding = 3) uniform usampler3D u_global_occluder;
layout(binding = 4) uniform sampler2DArray u_stbn_unitvec3;

layout(location = 0) out vec3 out_specular;

const int MAX_SPECULAR_SAMPLES = 1;
const int MAX_SPECULAR_DISTANCE = 400;
const float ROUGHNESS = 0.2;

uint rnd_seed = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * 4097U + uint(frame_counter);
int normal_rand_state = int(rnd_seed) % 64;
float linear_depth = texture(u_linear_depth, uv).r;
vec3 camera_pos = uInvView[3].xyz;
vec3 size = textureSize(u_global_occluder, 0);
Box bbox = Box(vec3(0.0), vec3(size));

vec3 reflect(vec3 I, vec3 N) {
    return I - 2.0 * dot(N, I) * N;
}

vec3 compute_specular_lighting(vec3 pos, vec3 normal) {
    vec3 ray_start = pos + normal * 0.1;

    // Specular Lighting
    vec3 specular = vec3(0.0);
    for (int spec_sample = 0; spec_sample < MAX_SPECULAR_SAMPLES; spec_sample += 1) {
        vec3 reflection_vec = reflect(normalize(pos - camera_pos), normal);
        vec3 random_normal = generate_random_stbn_unitvec3(u_stbn_unitvec3, normal_rand_state) * ROUGHNESS * ROUGHNESS;
        vec3 reflection_jittered = normalize(reflection_vec + random_normal);
        Ray occ_ray = Ray(ray_start, reflection_jittered);
        Hit occ_hit = dda(occ_ray, MAX_SPECULAR_DISTANCE, u_global_occluder, bbox);
        if (!occ_hit.hit) {
            specular += skyColor(occ_ray.direction, sun_direction);
        } else {
            // TODO(david): We could take a screen space sample here from the last frame's final texture, also use rejection
        }
    }
    return specular * 0.1 / MAX_SPECULAR_SAMPLES;
}

void main() {
    Ray camera_ray = compute_camera_ray(uv, uInvProjection, uInvView, 0, 0.0);
    float depth = texture(u_depth, uv).r;
    if (depth == 1.0) {
        return;
    }
    vec3 normal = texture(u_normal, uv).rgb;
    vec3 pos = camera_ray.origin + camera_ray.direction * linear_depth;
    vec3 color = compute_specular_lighting(pos, normal);

    out_specular = color;
}
#endif
