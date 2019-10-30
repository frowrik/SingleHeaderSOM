
#include <stdint.h>

#define    SingleHeaderSOM_MALLOC(Size)         Core->Memory_Heap_Get(Size)     // Set memory alloc fuction
#define    SingleHeaderSOM_FREE(Addr)           Core->Memory_Heap_Free(Addr)    // Set memory free fuction
#define    SingleHeaderSOM_ASSERT               DASSERT                         // Set assert function
#define    SingleHeaderSOM_FLOATRANDFUNCT()     RandFloatPlus(1.0)              // Set float random function
#define    SingleHeaderSOM_TEXPMIN              0.0000001                       // min time exp result

class SingleHeaderSOM {
public:
    SingleHeaderSOM() {
        // Nop, not used
    }
    
    SingleHeaderSOM(
        unsigned int MapWidth,
        unsigned int MapHeight,
        unsigned int VectorLen,
        float Radius,
        float Tsteps,
        float Lernrate) 
    {
        SingleHeaderSOM_ASSERT(MapWidth >= 1);
        SingleHeaderSOM_ASSERT(MapHeight >= 1);
        SingleHeaderSOM_ASSERT(VectorLen >= 1);

        Map_Width           = MapWidth;
        Map_Height          = MapHeight;
        Map_Radius          = Radius;
        Map_Tsteps          = Tsteps;
        Map_Lernrate        = Lernrate;
        Map_Vector_Count    = VectorLen;
        Map_Weights_Count   = Map_Width * Map_Height * Map_Vector_Count;
        Map_Weights         = (float*)SingleHeaderSOM_MALLOC(Map_Weights_Count * sizeof(float));
        
        Reset();
    }

    ~SingleHeaderSOM() {
        if (Map_Weights) SingleHeaderSOM_FREE(Map_Weights);
        Map_Weights = nullptr;
    }

    float* const GetMapData(unsigned int Map_X, unsigned int Map_Y) {
        SingleHeaderSOM_ASSERT(Map_X < Map_Width);
        SingleHeaderSOM_ASSERT(Map_Y < Map_Height);
        unsigned int Offset = (Map_X + Map_Y * Map_Height) * Map_Vector_Count;
        return &Map_Weights[Offset];
    }

    void    Reset() {
        for (unsigned int i = 0; i < Map_Weights_Count; i++) {
            Map_Weights[i] = SingleHeaderSOM_FLOATRANDFUNCT();
        }
        Map_t_step = 0;
    }

    void    TrainingStep(float* Vector) {
        unsigned int    Map_Win_X;
        unsigned int    Map_Win_Y;
        float            Map_Win_DistSQ;
        
        Step(Vector, &Map_Win_X, &Map_Win_Y, &Map_Win_DistSQ);

        float t_Exp         = (Map_t_step < Map_Tsteps) ? exp(-(float)Map_t_step / (float)Map_Tsteps) : SingleHeaderSOM_TEXPMIN;
        float Sigma         = Map_Radius    * t_Exp;
        float LerningRate   = Map_Lernrate  * t_Exp;
        float Sigma_Dist    = 1.0 / (2.0 * Sigma * Sigma);

        for (unsigned int y = 0, WIDX = 0; y < Map_Height; y++) {
            for (unsigned int x = 0; x < Map_Width; x++) {
                float Dist_X                    = (int)Map_Win_X - (int)x;
                float Dist_Y                    = (int)Map_Win_Y - (int)y;
                float Dist_2                    = Dist_X * Dist_X + Dist_Y * Dist_Y;
                float O_neighborhood            = exp(-Dist_2 * Sigma_Dist);
                float O_neighborhoodLerningRate = LerningRate * O_neighborhood;

                for (unsigned int i = 0; i < Map_Vector_Count; i++) {
                    Map_Weights[WIDX] += O_neighborhoodLerningRate * (Vector[i] - Map_Weights[WIDX]);
                    WIDX++;
                }
            }
        }

        Map_t_step++;
    }

    void    Step(float* Vector, unsigned int* Map_Win_X, unsigned int* Map_Win_Y, float* Map_Win_DistSQ) {
        SingleHeaderSOM_ASSERT(Map_Win_X);
        SingleHeaderSOM_ASSERT(Map_Win_Y);
        SingleHeaderSOM_ASSERT(Map_Win_DistSQ);
        
        unsigned int    Winner_X = 0;
        unsigned int    Winner_Y = 0;
        float           Winner_Dist_2 = FLT_MAX;

        for (unsigned int y = 0, WIDX = 0; y < Map_Height; y++) {
            for (unsigned int x = 0; x < Map_Width; x++) {
                float D = 0.0;

                for (unsigned int i = 0; i < Map_Vector_Count; i++) {
                    float dr = Map_Weights[WIDX++] - Vector[i];
                    D += dr * dr;
                }

                if (D < Winner_Dist_2) {
                    Winner_Dist_2 = D;
                    Winner_X = x;
                    Winner_Y = y;
                }
            }
        }

        *Map_Win_X          = Winner_X;
        *Map_Win_Y          = Winner_Y;
        *Map_Win_DistSQ     = Winner_Dist_2;
    }

private:
    unsigned int    Map_Width;
    unsigned int    Map_Height;
    float           Map_Radius;
    float           Map_Tsteps;
    float           Map_Lernrate;
    unsigned int    Map_Weights_Count;
    float*          Map_Weights = nullptr;
    unsigned int    Map_Vector_Count;
    unsigned int    Map_t_step;
};



