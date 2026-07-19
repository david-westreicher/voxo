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

in vec2 uv;

layout(binding = 0) uniform sampler2D input_tex;

uniform float filter_min;
uniform float filter_max;

out vec3 fragColor;

float rescale(float col) {
    return clamp((col - filter_min) / (filter_max - filter_min), 0.0, 1.0);
}

void main() {
    vec3 col = texture(input_tex, uv).rgb;
    fragColor = vec3(rescale(col.r), rescale(col.g), rescale(col.b));
}
#endif
