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
uniform bool is_material;
uniform int material_count;
uniform int material_row;

out vec4 fragColor;

void main() {
    // NOTE(david): materials always have 256 slots, slot 0 is the "empty-voxel-material"
    // material_count does not consider slot 0, we only want to show material[1:material_count+1]
    float matieral_coord_x = floor(mix(1, material_count + 1, uv.x));
    vec4 material = texelFetch(input_tex, ivec2(matieral_coord_x, material_row), 0).rgba;
    fragColor = vec4(0.0);
    if (is_material) {
        int row = int(gl_FragCoord.y);
        if (row == 0) {
            if (material.r > 0.0) {
                fragColor = vec4(vec3(material.r), 1.0);
            }
        }
        if (row == 1) {
            if (material.g > 0.0) {
                fragColor = vec4(vec3(material.g), 1.0);
            }
        }
        if (row == 2) {
            if (material.b > 0.0) {
                fragColor = vec4(vec3(material.b), 1.0);
            }
        }
        if (row == 3) {
            if (material.a > 0.0) {
                fragColor = vec4(vec3(material.a), 1.0);
            }
        }
        if (row == 4) {
            if (material.a < 0.0) {
                fragColor = vec4(vec3(abs(material.a)), 1.0);
            }
        }
    } else {
        fragColor = material;
    }
}
#endif
