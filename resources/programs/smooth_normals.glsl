#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 uv;

void main() {
    gl_Position = vec4(in_position, 1);
    uv = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER
#line 17

in vec2 uv;

uniform sampler2D input_texture;
uniform sampler2D depth_texture;
uniform mat4 uInvView;
uniform mat4 uInvProjection;

out vec4 fragColor;

const int RADIUS = 3;
const float SIGMA = 3.0;
const float MAX_DEPTH = 100.0;
vec2 texel_size = 1.0 / vec2(textureSize(input_texture, 0));

float gaussian(vec2 p) {
    return exp(-dot(p, p) / (2.0 * SIGMA * SIGMA));
}

vec3 reconstructWorldPos(vec2 uv, float depth)
{
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 ray = uInvProjection * vec4(ndc, 1.0, 1.0);
    vec3 view_dir = normalize(ray.xyz / ray.w);
    vec3 world_dir = normalize((uInvView * vec4(view_dir, 0.0)).xyz);
    vec3 camera_pos = uInvView[3].xyz;
    return camera_pos + world_dir * depth;
}

vec3 blur(vec2 uv) {
    vec3 color = vec3(0.0);
    float weight_sum = 0.0;
    float center_depth = texture(depth_texture, uv).r;
    vec3 center_normal = texture(input_texture, uv).rgb;
    if (center_depth >= MAX_DEPTH) {
        return center_normal;
    }
    vec3 center_normal_decoded = center_normal;
    vec3 center_pos = reconstructWorldPos(uv, center_depth);
    vec2 local_texel_size = texel_size * max(0, 1.0 - pow(center_depth / MAX_DEPTH, 0.2));

    for (int y = -RADIUS; y <= RADIUS; ++y)
    {
        for (int x = -RADIUS; x <= RADIUS; ++x)
        {
            vec2 p = vec2(x, y);
            vec2 sample_uv = uv + p * local_texel_size;
            float sample_depth = texture(depth_texture, sample_uv).r;
            vec3 sample_pos = reconstructWorldPos(sample_uv, sample_depth);
            float depth_diff = abs(sample_depth - center_depth);
            // reject concave and far samples
            if (dot((sample_pos - center_pos), center_normal_decoded) > 0.001 || depth_diff > 0.1) {
                continue;
            }
            float w = gaussian(p);
            color += texture(input_texture, sample_uv).rgb * w;
            weight_sum += w;
        }
    }

    return color / weight_sum;
}

void main() {
    fragColor = vec4(blur(uv), 1.0);
}
#endif
