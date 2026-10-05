#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "compaction.h"

#define ERRORCHECK 1
#define STREAM_COMPACTION 1
#define SORT_BY_MATERIAL 1
#define CUSTOM_COMPACTION 1  // 1 = our scan + partition (compaction.cu), 0 = thrust::partition
#define ANTIALIASING 1
#define RUSSIAN_ROULETTE 1
#define RR_MIN_DEPTH 3  // never roulette the first few bounces, these carry most of the light
#define MESH_BBOX_CULLING 1


// isAlive is a struct that becomes the Predicate for thrust::partition, which will call isAlive(dev_paths[i]). kind of a hack
struct isAlive
{
    __host__ __device__ bool operator()(const PathSegment& p) const
    {
        return p.remainingBounces > 0;
    }
};

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
// see allocation below for explanations
static int* dev_materialKeys = NULL;
static Triangle* dev_triangles = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;  // the camera is not moving
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    // for every ray, we'll store an int key for the material it hit, and then radix sort materialkeys:(paths, intersections) with thrust 
    // radix sorting like this will be faster than trying to sort on intersections:paths directly with a materialId comparison
    cudaMalloc(&dev_materialKeys, pixelcount * sizeof(int));

    // we are going to store one big triangles array, which will be shared across all meshes (if there are multiple).
    // the Geoms will index into this array with triStart/triCount. Scene stores it. this is also way cleaner than making every triangle its own geom.
    cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
    cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

    Compaction::init(pixelcount);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_materialKeys);
    cudaFree(dev_triangles);
    Compaction::free();

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/

// happens once at the beginning, so one thread per pixel
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // implement antialiasing by jittering the ray
        float fx = (float)x;
        float fy = (float)y;
        
        // shared 
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
#if ANTIALIASING
        // add some random noise to the primary ray direction
        thrust::uniform_real_distribution<float> u01(-0.5f, 0.5f);
        fx += u01(rng);
        fy += u01(rng);
#endif
        // in the regular path, we don't care about the distance to the virtual image plane
        // d = this pixel's spot on the virtual image plane at distance 1
        glm::vec3 d = cam.view
            - cam.right * cam.pixelLength.x * (fx - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * (fy - (float)cam.resolution.y * 0.5f);

        segment.ray.direction = glm::normalize(d);

        // thin lens drpth of field.
        // we're essentially running the opposite of computer vision logic here: 
        // in a real camera, for a given focal length and d_i, this determines the d_o that will result in a sharp image. here, we're setting it directly in the scene.
        if (cam.lensRadius > 0.0f)
        {
            thrust::uniform_real_distribution<float> uLens(0, 1);

            // move the origin to random point on the (circular) lens disk. sqrt so it's uniform over the area
            float r = cam.lensRadius * sqrtf(uLens(rng));
            float phi = TWO_PI * uLens(rng);
            float lx = r * cosf(phi);
            float ly = r * sinf(phi);

            segment.ray.origin = cam.position + cam.right * lx + cam.up * ly;
            
            // now adjust the direction
            // where is the image sharp? its where the image plane is pushed out to focalDistance
            glm::vec3 pFocus = cam.position + cam.focalDistance * d;

            segment.ray.direction = glm::normalize(pFocus - segment.ray.origin);
        }

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Triangle* triangles,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;  // the intersection test already tests for outside. we can use it directly for refraction in scatterRay
        bool hit_outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms
        // we can accelerate this dramatically using a BVH
        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            // set intersection point, normal, and outside
            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == MESH)
            {
                // we'll iterate through all of the triangles within this
                t = meshIntersectionTest(geom, pathSegment.ray, triangles, MESH_BBOX_CULLING, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                // make sure to overwite hit_outside
                hit_outside = outside;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].outside = hit_outside;
        }
    }
}

__global__ void kernGetMaterialKeys(int num_paths, ShadeableIntersection* intersections, int* keys)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {   
        // we'll make misses have a key of -1 so they can be sorted together at the front of the array
        keys[idx] = intersections[idx].t < 0.0f ? -1 : intersections[idx].materialId;
    }
}

__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    // because we've radix sorted the paths by material, nearby threads will have the same intersection.materialId
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        PathSegment& segment = pathSegments[idx];

        // already dead
        if (segment.remainingBounces <= 0) return;

        // missed everything
        if (intersection.t < 0.0f)
        {
            segment.color = glm::vec3(0.0f);
            segment.remainingBounces = 0;
            return;
        }

        // here
        Material material = materials[intersection.materialId];

        // hit a light
        if (material.emittance > 0.0f)
        {
            segment.color *= material.color * material.emittance;
            segment.remainingBounces = 0;
            return;
        }

        // seed with depth too so each bounce gets different randoms
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
        glm::vec3 intersect = segment.ray.origin + intersection.t * segment.ray.direction;

        scatterRay(segment, intersect, intersection.surfaceNormal, intersection.outside, material, rng);
        segment.remainingBounces--;

#if RUSSIAN_ROULETTE
        if (depth >= RR_MIN_DEPTH && segment.remainingBounces > 0)
        {
            thrust::uniform_real_distribution<float> u01(0, 1);
            // remember, color is the throughput of the path. this provides a more unbiased way to terminate paths to boost efficiency.
            // dim paths are likely to die, but bright paths can still die eventually
            float p = glm::min(glm::max(segment.color.r, glm::max(segment.color.g, segment.color.b)), 0.95f);
            
            // killed
            if (u01(rng) >= p)
            {
                segment.color = glm::vec3(0.0f);
                segment.remainingBounces = 0;
            }

            // alive
            else
            {
                // need to keep the expected value the same. p*(C/p) + (1-p)*0 = C
                segment.color /= p;
            }
        }
#endif

        // out of bounces without hitting a light
        if (segment.remainingBounces <= 0)
        {
            segment.color = glm::vec3(0.0f);
        }
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;


        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

#if SORT_BY_MATERIAL
        // int keys so thrust can radix sort
        kernGetMaterialKeys<<<numblocksPathSegmentTracing, blockSize1d>>>(num_paths, dev_intersections, dev_materialKeys);
        checkCUDAError("get material keys");
        thrust::sort_by_key(thrust::device, dev_materialKeys, dev_materialKeys + num_paths,
            thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections)));
#endif

        // shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
        //     iter,
        //     num_paths,
        //     dev_intersections,
        //     dev_paths,
        //     dev_materials
        // );

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );
        // after this point, some pathsegments might have been terminated. we can stream compact them out
        checkCUDAError("shade one bounce");

#if STREAM_COMPACTION
#if CUSTOM_COMPACTION
        // uses shared memory
        num_paths = Compaction::partitionPaths(num_paths, dev_paths);
#else
        // partition into alive | dead. the predicate here is isAlive, which checks if remainingBounces > 0
        // we can pass in thrust::device as an argument instead of making thrust::device_vectors or anything
        PathSegment* dev_alive_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, isAlive());
        num_paths = dev_alive_end - dev_paths;
#endif
#endif

        // either every path has terminated or we've reached the maximum depth
        iterationComplete = num_paths == 0 || depth >= traceDepth;

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);  // whole image! not just num_paths anymore

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    if (pbo) sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);  // null in headless mode

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
