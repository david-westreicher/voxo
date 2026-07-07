#version 330

#if defined VERTEX_SHADER

in vec3 in_position;
in vec2 in_texcoord_0;

out vec2 vUV;

void main() {
    gl_Position = vec4(in_position, 1);
    vUV = in_texcoord_0;
}

#elif defined FRAGMENT_SHADER

uniform vec3 uCameraPos;
uniform mat4 uInvView;
uniform mat4 uInvProjection;
uniform sampler3D u_voxel_data;

in vec2 vUV;
out vec4 fragColor;

const int MAX_STEPS = 50;
float size = 10.0;
vec3 boxMin = vec3(-(size * 0.5 + 0.5));
vec3 boxMax = vec3(size * 0.5 - 0.5);

float voxelmap(vec3 p) // voxel map
{
    vec3 local_coord = (p - boxMin) / (boxMax - boxMin); // Normalize to [0,1] range
    return texture(u_voxel_data, local_coord).r - 0.8;
}

bool is_inside_box(vec3 p) {
    return all(greaterThanEqual(p, boxMin)) &&
           all(lessThanEqual(p, boxMax));
}

// Code modified from: https://www.shadertoy.com/view/X3BXDd
// Improved Branchless Voxel DDA
void mainImage( out vec4 fragColor, in vec2 uv, in vec3 pos, in vec3 rayDir)
{
    vec3 color = vec3(0);
    
    // DDA setup

    vec3 map = floor(pos);           // integer cell coordinate of initial / current cell

    vec3 stepDir=vec3(0);            // step sign +/- 1
    vec3 sideDist=vec3(9e9);         // initial distance to cell sides, then relative difference between traveled sides
    
    // Note the use of un-normalized rayDir here! using normalized rayDir or length() here gives ragged edges artifact!
    vec3 deltaDist = 1./abs(rayDir); // ray length required to step from one cell border to the next in x, y and z directions
    
    float side=0.;

    vec3 S = step(0., rayDir); // S is rayDir non-negative? 0 / 1
    stepDir = 2.*S-1.;
    
    // if 1./abs(rayDir[i]) is inf, then rayDir[i] is 0., but then S = step(0., rayDir[i]) is 1
    // so S cannot be 0. while deltaDist is inf, and stepDir * fract(pos) can never be 1.
    // Therefore we should not have to worry about getting NaN here :)
    
    sideDist = (S-stepDir * fract(pos)) * deltaDist;   // alternative: //sideDist = (S-stepDir * (pos - map)) * deltaDist;
    
    // DDA marching 
    
    int i = 0;
    bool has_entered = is_inside_box(map);
    for(;i < MAX_STEPS; i++)
    {
        // Decide which way to go!
        vec4 conds = step(sideDist.xxyy, sideDist.yzzx); // same as vec4(sideDist.xxyy <= sideDist.yzzx);
        
        // This mimics the if, elseif and else clauses
        // * is 'and', 1.-x is negation
        vec3 cases = vec3(0);
        cases.x = conds.x * conds.y;                 // if       x dir
        cases.y = (1.-cases.x) * conds.z * conds.w;  // else if  y dir
        cases.z = (1.-cases.x) * (1.-cases.y);       // else     z dir
        
        // usually would have been:     sideDist += cases * deltaDist;
        // but this gives NaN when  cases[i] * deltaDist[i]  becomes  0. * inf 
        // This gives NaN result in a component that should not have been affected,
        // so we instead give negative results for inf by mapping 'cases' to +/- 1
        // and then clamp negative values to zero afterwards, giving the correct result! :)
        sideDist += max((2.*cases-1.) * deltaDist, 0.);
        
        map += cases * stepDir;

        if(is_inside_box(map) && !has_entered) {
            has_entered = true;
        }
        if(has_entered && !is_inside_box(map)) {
            fragColor = vec4(0.0);
            return;
        }

        
        if(has_entered && voxelmap(map) > 0.) // Did we hit anything? if so, we are done!
        {
            side = cases.y + 2. * cases.z;
            break;
        }
    }

    if (i == MAX_STEPS) {
        fragColor = vec4(0.0);
        return;
    }

    color[int(side)] = 1.; // voxel face debug
    color = color * (0.25 + 0.5 * float(float(mod(map.x,2.) != mod(map.y,2.))!=mod(map.z,2.)) );
    color = pow(clamp(color, 0., 1.), vec3(1./2.2));
    fragColor = vec4(color,0.0);
}


void main() {
    vec2 ndc = vUV * 2.0 - 1.0;

    vec4 clip = vec4(ndc, -1.0, 1.0);
    vec4 eye = uInvProjection * clip;
    eye = vec4(eye.xy, -1.0, 0.0);

    vec3 rd = normalize((uInvView * eye).xyz);
    vec3 ro = uCameraPos;

    mainImage(fragColor,ndc,ro,rd);
}

#endif
