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
#line 15

in vec2 uv;

layout(binding = 0) uniform sampler2D u_input_texture;
uniform float exposure = 4.0;
uniform float knee_factor = 1.5;

layout(location = 0) out vec3 fragColor;

const vec2 offsets[] = vec2[](
        vec2(0.0, 0.0),
        vec2(-1.0, -1.0),
        vec2(-1.0, 1.0),
        vec2(1.0, -1.0),
        vec2(1.0, 1.0)
    );
const float knee = exposure * 0.5;

vec2 texel_size = 1.0 / textureSize(u_input_texture, 0);

vec3 apply_threshold(vec3 col, inout float luminance) {
    luminance = dot(col, vec3(0.2126, 0.7152, 0.0722));
    float soft = clamp(luminance - exposure + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee);
    float contribution = max(soft, luminance - exposure);
    contribution /= max(luminance, 1e-5);
    return col * contribution;
}

void main() {
    vec3 color = vec3(0.0);
    float weight_sum = 0.0;

    // Reduce firefly flickering by taking a weighted average over close by texels
    // https://catlikecoding.com/unity/tutorials/custom-srp/hdr/
    for (int i = 0; i < 5; i++) {
        vec3 c = texture(u_input_texture, uv + offsets[i] * texel_size * 2.0).rgb;
        float luminance;
        c = apply_threshold(c, luminance);
        float w = 1.0 / (luminance + 1.0);
        color += c * w;
        weight_sum += w;
    }
    color /= weight_sum;
    fragColor = color;
}
#endif
