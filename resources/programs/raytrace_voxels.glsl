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
uniform usampler3D u_voxel_data;
uniform sampler2D u_palette_data;

in vec2 vUV;
out vec4 fragColor;

vec3 size = ceil(textureSize(u_voxel_data, 0) * 0.5) * 2.0; // NOTE(david): hack to make odd dimensions work
float inv_palette_size = 1.0 / (textureSize(u_palette_data, 0).r - 1.0);
int MAX_STEPS = int(max(size.x, max(size.y, size.z)))*3;

vec3 boxMin = -size * 0.5;
vec3 boxMax = size * 0.5;
vec3 lightPos = vec3(MAX_STEPS);


uint voxelmap(vec3 p)
{
    vec3 local_coord = (p - boxMin + 0.5) / vec3(size);
    return texture(u_voxel_data, local_coord).r;
}

bool is_inside_box(vec3 p) {
    return all(greaterThanEqual(p, boxMin)) &&
           all(lessThan(p, boxMax));
}

vec3 diffuseLight(vec3 normal, vec3 lightDir, vec3 lightColor, vec3 albedo)
{
    float NdotL = max(dot(normalize(normal), normalize(lightDir)), 0.0);
    return albedo * lightColor * NdotL;
}

vec3 phongLight(
    vec3 normal,
    vec3 lightDir,
    vec3 viewDir,
    vec3 lightColor,
    vec3 albedo,
    float shininess
)
{
    vec3 N = normalize(normal);
    vec3 L = normalize(lightDir);
    vec3 V = normalize(viewDir);

    // Diffuse (Lambert)
    float NdotL = max(dot(N, L), 0.0);
    vec3 diffuse = albedo * lightColor * NdotL;

    // Specular (Phong)
    vec3 R = reflect(-L, N);
    float spec = pow(max(dot(R, V), 0.0), shininess);
    vec3 specular = lightColor * spec;

    return diffuse + specular;
}

bool intersectAABB(
    vec3 rayOrigin,
    vec3 rayDir,
    vec3 boxMin,
    vec3 boxMax,
    out float tHit
)
{
    vec3 invDir = 1.0 / rayDir;

    vec3 t0 = (boxMin - rayOrigin) * invDir;
    vec3 t1 = (boxMax - rayOrigin) * invDir;

    vec3 tMin = min(t0, t1);
    vec3 tMax = max(t0, t1);

    float tNear = max(max(tMin.x, tMin.y), tMin.z);
    float tFar  = min(min(tMax.x, tMax.y), tMax.z);

    // No intersection, or box is behind ray
    if (tNear > tFar || tFar < 0.0)
        return false;

    tHit = max(tNear, 0.0);
    return true;
}

// Code modified from: https://www.shadertoy.com/view/X3BXDd
// Improved Branchless Voxel DDA
void mainImage( out vec4 fragColor, in vec3 pos, in vec3 rayDir)
{
    
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
            discard;
        }

        if(has_entered && voxelmap(map) > 0.) // Did we hit anything? if so, we are done!
        {
            side = cases.y + 2. * cases.z;
            break;
        }
    }

    vec3 albedo = vec3(0.0);
    albedo[int(side)] = 1.; // voxel face debug
    albedo = albedo * (0.25 + 0.5 * float(float(mod(map.x,2.) != mod(map.y,2.))!=mod(map.z,2.)) );

    //vec2 palette_coord = vec2(0.5,0.5);
    vec2 palette_coord = vec2(float(voxelmap(map)) * inv_palette_size);
    albedo = texture(u_palette_data, palette_coord).rgb;

    vec3 n = vec3(0.0);
    n[int(side)] = -1. * sign(rayDir[int(side)]); // voxel face debug
    vec3 p = map + .5 - stepDir*.5; // Point on axis plane

    // Solve ray plane intersection equation: dot(n, ro + t * rd - p) = 0.
    // for t :
    float t = (dot(n, p - pos)) / dot(n, rayDir);
    vec3 hit = pos + rayDir * t;

    vec3 normal = vec3(0);
    normal[int(side)] = -1. * sign(rayDir[int(side)]); // voxel face debug
    vec3 L = normalize(lightPos - hit); // direction to light
    vec3 V = normalize(pos - hit);

    vec3 color = phongLight(
        normal,
        L,
        V,
        vec3(1.0),              // light color
        albedo,                 // material/albedo
        64.0                    // shininess
    );

    fragColor = vec4(color, 1.0);
}


void main() {
    vec2 ndc = vUV * 2.0 - 1.0;
    vec4 clip = vec4(ndc, -1.0, 1.0);
    vec4 eye = uInvProjection * clip;
    eye = vec4(eye.xy, -1.0, 0.0);

    vec3 rd = normalize((uInvView * eye).xyz);
    vec3 ro = uCameraPos;

    float t;
    if (intersectAABB(ro, rd, boxMin, boxMax, t))
    {
        vec3 hitPos = ro + (t-0.01) * rd;
        mainImage(fragColor, hitPos, rd);
    } else {
        discard;
    }
}

#endif
