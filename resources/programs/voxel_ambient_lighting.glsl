#version 430

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

uniform mat4 uView;
uniform mat4 uProjection;
uniform mat4 uInvView;
uniform mat4 uInvProjection;
uniform int frame_counter;
uniform int max_occ_samples;

layout(binding = 0) uniform sampler2D u_normal;
layout(binding = 1) uniform sampler2D u_depth;
layout(binding = 2) uniform sampler2D u_linear_depth;
layout(binding = 3) uniform usampler3D u_global_occluder;
layout(binding = 4) uniform sampler2DArray u_stbn_normals;
layout(binding = 5) uniform sampler2DArray u_stbn_vec3;

layout(location = 0) out vec3 out_irradiance;

const int MAX_OCC_DISTANCE = 300;

uint rnd_seed = uint(gl_FragCoord.x) + uint(gl_FragCoord.y) * 4097U + uint(frame_counter);
int normal_rand_state = int(rnd_seed) % 64;
int jitter_pos_state = int(rnd_seed) % 64;

mat4 projectionview = uProjection * uView;
float linear_depth = texture(u_linear_depth, uv).r;
vec3 camera_pos = uInvView[3].xyz;
vec3 size = textureSize(u_global_occluder, 0);
Box bbox = Box(vec3(0.0), vec3(size));

vec3 compute_ambient_lighting(vec3 pos, vec3 normal, Pcg32State rnd) {
    vec3 ray_start = pos + normal * 0.5;

    // Ambient Lighting
    vec3 ambient = vec3(0.0);
    for (int occ_sample = 0; occ_sample < max_occ_samples; occ_sample += 1) {
        vec3 jitter_point = (generate_random_stbn_vec3(u_stbn_vec3, jitter_pos_state) - 0.5);
        vec3 jitter = jitter_point - normal * dot(jitter_point, normal);
        Ray occ_ray = Ray(ray_start + jitter, generate_random_cosine_weighted_normal(normal, u_stbn_normals, normal_rand_state));
        Hit occ_hit = sparse_raymarch(occ_ray, MAX_OCC_DISTANCE, u_global_occluder, bbox, 2);
        if (!occ_hit.hit) {
            ambient += skyColor(occ_ray.direction, vec3(0, -1, 0));
        }
        // TODO(david): We could take a screen space sample here from the last frame's irradiance texture, also use rejection
    }
    return ambient / max_occ_samples;
}

void main() {
    Pcg32State rnd = pcg_srandom(rnd_seed);

    Ray camera_ray = compute_camera_ray(uv, uInvProjection, uInvView, 0, 0.0);
    float depth = texture(u_depth, uv).r;
    if (depth == 1.0) {
        return;
    }
    vec3 normal = texture(u_normal, uv).rgb;
    vec3 pos = camera_ray.origin + camera_ray.direction * linear_depth;
    vec3 color = compute_ambient_lighting(pos, normal, rnd);

    out_irradiance = color;
}
#endif
