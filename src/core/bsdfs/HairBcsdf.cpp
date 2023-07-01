#include "HairBcsdf.hpp"

#include "sampling/PathSampleGenerator.hpp"

#include "bsdfs/Fresnel.hpp"

#include "math/GaussLegendre.hpp"
#include "Microfacet.hpp"
#include "io/JsonObject.hpp"

namespace Tungsten {


HairBcsdf::HairBcsdf()
: _scaleAngleDeg(2.0f),
  _melaninRatio(0.5f),
  _melaninConcentration(0.25f),
  _overridesSigmaA(false),
  _sigmaA(0.0f),
  _roughness(0.1f)
{
    alpha_r = 1.5;
    alpha_tt  = alpha_r * 0.5;
    alpha_trt = alpha_r * 2;
    _lobes = BsdfLobes(BsdfLobes::GlossyLobe | BsdfLobes::AnisotropicLobe);
}

// Modified Bessel function of the first kind
float HairBcsdf::I0(float x)
{
    float result = 1.0f;
    float xSq = x*x;
    float xi = xSq;
    float denom = 4.0f;
    for (int i = 1; i <= 10; ++i) {
        result += xi/denom;
        xi *= xSq;
        denom *= 4.0f*float((i + 1)*(i + 1));
    }
    return result;
}

float HairBcsdf::logI0(float x)
{
    if (x > 12.0f)
        // More stable evaluation of log(I0(x))
        // See also https://publons.com/discussion/12/
        return x + 0.5f*(std::log(1.0f/(TWO_PI*x)) + 1.0f/(8.0f*x));
    else
        return std::log(I0(x));
}

// Standard normalized Gaussian
float HairBcsdf::g(float beta, float theta)
{
    return std::exp(-theta*theta/(2.0f*beta*beta))/(std::sqrt(2.0f*PI)*beta);
}

// Wrapped Gaussian "detector", computed as an infinite sum of Gaussians
// Approximated with a finite sum for obvious reasons.
// Note: This is merely to reproduce the paper. You could
// (and probably should) replace this with some other Gaussian-like
// function that can be analytically normalized and sampled
// over the [-Pi, Pi] domain. The Gaussian cannot, hence this
// slightly awkward and expensive evaluation.
float HairBcsdf::D(float beta, float phi)
{
    float result = 0.0f;
    float delta;
    float shift = 0.0f;
    do {
        delta = g(beta, phi + shift) + g(beta, phi - shift - TWO_PI);
        result += delta;
        shift += TWO_PI;
    } while (delta > 1e-4f);
    return result;
}

// Computes the exitant azimuthal angle of the p'th perfect specular
// scattering event, derived using Bravais theory
// See the paper "Light Scattering from Human Hair Fibers" for details
float HairBcsdf::Phi(float gammaI, float gammaT, int p)
{
    return 2.0f*p*gammaT - 2.0f*gammaI + p*PI;
}

// The following two functions are the guts of the azimuthal scattering function,
// following the paper. They are only provided for reference - the actual
// runtime evaluation uses precomputed versions of these functions
// sampled into a 2D table. Additionally, the precomputation turns these functions inside out
// to cache values that are constant across successive evaluations, and does not
// use these functons directly.

// Special case for the R lobe
float HairBcsdf::NrIntegrand(float beta, float halfWiDotWo, float phi, float h) const
{
    float gammaI = std::asin(clamp(h, -1.0f, 1.0f));
    float deltaPhi = phi + 2.0f*gammaI;
    deltaPhi = std::fmod(deltaPhi, TWO_PI);
    if (deltaPhi < 0.0f)
        deltaPhi += TWO_PI;

    return D(beta, deltaPhi)*Fresnel::dielectricReflectance(1.0f/Eta, halfWiDotWo);
}

Vec3f HairBcsdf::NpIntegrand(float beta, float cosThetaD, float phi, int p, float h) const
{
    float iorPrime = std::sqrt(Eta*Eta - (1.0f - cosThetaD*cosThetaD))/cosThetaD;
    float cosThetaT = std::sqrt(1.0f - (1.0f - cosThetaD*cosThetaD)*sqr(1.0f/Eta));
    Vec3f sigmaAPrime = _sigmaA/cosThetaT;

    float gammaI = std::asin(clamp(h, -1.0f, 1.0f));
    float gammaT = std::asin(clamp(h/iorPrime, -1.0f, 1.0f));
    // The correct internal path length (the one in d'Eon et al.'s paper
    // as well as Marschner et al.'s paper is wrong).
    // The correct factor is also mentioned in "Light Scattering from Filaments", eq. (20)
    float l = 2.0f*std::cos(gammaT);

    float f = Fresnel::dielectricReflectance(1.0f/Eta, cosThetaD*trigInverse(h));
    Vec3f T = std::exp(-sigmaAPrime*l);
    Vec3f Aph = (1.0f - f)*(1.0f - f)*T;
    for (int i = 1; i < p; ++i)
        Aph *= f*T;

    float deltaPhi = phi - Phi(gammaI, gammaT, p);
    deltaPhi = std::fmod(deltaPhi, TWO_PI);
    if (deltaPhi < 0.0f)
        deltaPhi += TWO_PI;

    return Aph*D(beta, deltaPhi);
}

// Rough longitudinal scattering function with variance v = beta^2
float HairBcsdf::M(float v, float sinThetaI, float sinThetaO, float cosThetaI, float cosThetaO) const
{
    float a = cosThetaI*cosThetaO/v;
    float b = sinThetaI*sinThetaO/v;

    if (v < 0.1f)
        // More numerically stable evaluation for small roughnesses
        // See https://publons.com/discussion/12/
        return std::exp(-b + logI0(a) - 1.0f/v + 0.6931f + std::log(1.0f/(2.0f*v)));
    else
        return std::exp(-b)*I0(a)/(2.0f*v*std::sinh(1.0f/v));
}

// Returns sinThetaO
float HairBcsdf::sampleM(float v, float sinThetaI, float cosThetaI, float xi1, float xi2) const
{
    // Version from the paper (very unstable)
    //float cosTheta = v*std::log(std::exp(1.0f/v) - 2.0f*xi1*std::sinh(1.0f/v));
    // More stable version from "Numerically stable sampling of the von Mises Fisher distribution on S2 (and other tricks)"
    float cosTheta = 1.0f + v*std::log(xi1 + (1.0f - xi1)*std::exp(-2.0f/v));
    float sinTheta = trigInverse(cosTheta);
    float cosPhi = std::cos(TWO_PI*xi2);

    return -cosTheta*sinThetaI + sinTheta*cosPhi*cosThetaI;
}

void HairBcsdf::fromJson(JsonPtr value, const Scene &scene)
{
    Bsdf::fromJson(value, scene);
    value.getField("scale_angle", _scaleAngleDeg);
    value.getField("melanin_ratio", _melaninRatio);
    value.getField("melanin_concentration", _melaninConcentration);
    _overridesSigmaA = value.getField("sigma_a", _sigmaA);
    value.getField("roughness", _roughness);
}

rapidjson::Value HairBcsdf::toJson(Allocator &allocator) const
{
    JsonObject result{Bsdf::toJson(allocator), allocator,
        "type", "hair",
        "scale_angle", _scaleAngleDeg,
        "roughness", _roughness
    };

    if (_overridesSigmaA)
        result.add("sigma_a", _sigmaA);
    else
        result.add("melanin_ratio", _melaninRatio,
                   "melanin_concentration", _melaninConcentration);

    return result;
}
#include "Microfacet.hpp"

Vec3f wm_local(const Vec3f & wh,const Vec3f & wm){
   TangentFrame wmFrame;
    wmFrame.normal = wm;
    wmFrame.tangent = Vec3f(0.f, 1.f, 0.f).cross(wm);
    wmFrame.bitangent = wmFrame.normal.cross(wmFrame.tangent);
    return wmFrame.toLocal(wh);
}
template<class t>
inline  float dot(t a,t b)
{
    return a.dot(b);
}
    float getPhi(Vec3f a){
        return std::atan2(a.x(),a.z());
    }
static Vec3f eval_dielectric(Vec3f wi,Vec3f wo,Vec3f wh,Vec3f wm,float alpha,float eta = 1.55){
    if( dot(wo,wm) < 0)
    {
        return Vec3f(1.f,0.f,0.f);
        auto d1 = dot(wi,wm);
        auto d2= dot(wo,wm);;
        return Vec3f(0.f);
    }
    if(  dot(wh,wi)>0){
     //   return Vec3f(0.f,1.f,0.f);
        return Vec3f(0.f);

    }
    if(dot(wm,wh)<0)
    {
        return Vec3f(0.f,0.f,1.f);
        return Vec3f(0.f);}
    float whDotIn =  dot(wh,wi);
    float whDotOut = dot(wh,wo);
    float sqrtDeom = eta * whDotOut  +  whDotIn;
    auto d= Tungsten::Microfacet::D(Microfacet::GGX,alpha,wm_local(wh,wm));
    auto G = Tungsten::Microfacet::G(Microfacet::GGX,alpha,wm_local(wo,wm), wm_local(wi,wm),wm);
    return Vec3f(d) * G  *
std::abs(
        whDotIn * whDotOut  /
(wo.dot(wm) * sqrtDeom * sqrtDeom));
}
std::pair<float,float> sincos(float  angle)  {
    return {sin(angle), cos(angle)};
}
template<class T>
inline  T select(bool mask,T a,T b){
    return mask?a:b;
}
    template<class T>
    inline  T normalize(T a){
        return a.normalized();
    }
std::tuple<float, float, float, float> fresnel(float cos_theta_i, float eta) {
    bool outside_mask = cos_theta_i > 0;
    float  rcp_eta = 1.f/eta,
            eta_it = select(outside_mask, eta, rcp_eta),
            eta_ti = select(outside_mask, rcp_eta, eta);
    float  cosThetaT;
    auto r = Fresnel::dielectricReflectance(1/eta,cos_theta_i,cosThetaT);
    return {r,cosThetaT,eta_it,eta_ti};
}
    inline Vec3f Reflect(const Vec3f &wo, const Vec3f &n) {
        return normalize(-wo + 2 * dot(wo, n) * n);
    }
inline Vec3f sphDir(float phi,float theta){
    auto [sin_theta, cos_theta] = sincos(theta);
    auto [sin_gamma,   cos_gamma]   = sincos(phi);
    return Vec3f (sin_gamma * cos_theta, sin_theta, cos_gamma * cos_theta);
}

static inline Vec3f refract(const Vec3f &out, Vec3f &wh, float cosThetaT,
                            float eta) {
    auto whDotOut = dot(out,wh);
    return  (eta * whDotOut - (whDotOut>0?1:-1)*cosThetaT)* wh - eta* out ;
}

bool useGgx  =true;
Vec3f HairBcsdf::eval(const SurfaceScatterEvent &event) const
{
  //  return event.wo.y()>0?Vec3f(1):Vec3f(0.f);
    if (!event.requestedLobe.test(BsdfLobes::GlossyLobe))
        return Vec3f(0.0f);

    float sinThetaI = event.wi.y();
    float sinThetaO = event.wo.y();
    float cosThetaO = trigInverse(sinThetaO);
    float thetaI = std::asin(clamp(sinThetaI, -1.0f, 1.0f));
    float thetaO = std::asin(clamp(sinThetaO, -1.0f, 1.0f));
    float thetaD = (thetaO - thetaI)*0.5f;
    float cosThetaD = std::cos(thetaD);

    float phi = std::atan2(event.wo.x(), event.wo.z());
    if (phi < 0.0f)
        phi += TWO_PI;


    // Lobe shift due to hair scale tilt, following the values in
    // "Importance Sampling for Physically-Based Hair Fiber Models"
    // rather than the earlier paper by Marschner et al. I believe
    // these are slightly more accurate.
    float thetaIR   = thetaI - 2.0f*_scaleAngleRad;
    float thetaITT  = thetaI +      _scaleAngleRad;
    float thetaITRT = thetaI + 4.0f*_scaleAngleRad;

  //  return Vec3f(M(_vR,   std::sin(thetaIR),   sinThetaO, std::cos(thetaIR),   cosThetaO)) *   _nR->eval(phi, cosThetaD);
    // Evaluate longitudinal scattering functions
    float MR   = M(_vR,   std::sin(thetaIR),   sinThetaO, std::cos(thetaIR),   cosThetaO);
    float MTT  = M(_vTT,  std::sin(thetaITT),  sinThetaO, std::cos(thetaITT),  cosThetaO);
    float MTRT = M(_vTRT, std::sin(thetaITRT), sinThetaO, std::cos(thetaITRT), cosThetaO);

  //  return  MR*  _nR->eval(phi, cosThetaD);
  //  return MR*  _nR->eval(phi, cosThetaD);
    
    if(!useGgx)
        return  _nTT->eval(phi, cosThetaD);
//    return   MR*  _nR->eval(phi, cosThetaD)
//         +  MTT* _nTT->eval(phi, cosThetaD)
//         + MTRT*_nTRT->eval(phi, cosThetaD);

    auto Nr = _nR->eval(phi, cosThetaD),Ntt = _nTT->eval(phi, cosThetaD),Ntrt = _nTRT->eval(phi, cosThetaD);

    auto wi = event.wo,wo = event.wi;
    auto h = event.info->uv.y()*2-1;
    h = 2 * h -1;
    //return Vec3f(h<0?0:1);
    float gammaI = std::asin(clamp(h, - 1.0f, 1.0f));
    float _eta = 1.55;
    float iorPrime = std::sqrt(_eta * _eta - ( 1.0f - cosThetaD * cosThetaD )) / cosThetaD;
    float gammaT = std::asin(clamp(h / iorPrime, - 1.0f, 1.0f));
    auto phiO = std::atan2(wo.x(),wo.z());
    auto wm_r_phi = phiO - gammaI;
  //  wm_r_phi = phiO  + gammaI - PI;
    auto wm_tt_phi = gammaI - (PI - 2 * gammaT);
    auto wm_trt_phi = gammaI - 2 *(PI - 2 * gammaT);
    wm_tt_phi = PI + (wm_r_phi - (PI - 2 * gammaT));
    wm_trt_phi = PI+(wm_tt_phi+PI - (PI - 2 * gammaT));


    auto wm_r_theta =   +   2 * _scaleAngleRad;
    auto wm_tt_theta =(  - _scaleAngleRad);
    auto wm_trt_theta =(+ 4 * _scaleAngleRad);

//    if(wo.y<0)
//    {
//        wm_r_theta = - wm_r_theta;
//        wm_tt_theta = -wm_tt_theta;
//        wm_trt_theta = - wm_trt_theta;
//    }

    auto wm_r = sphDir(wm_r_phi,wm_r_theta);
    auto wm_tt = sphDir(wm_tt_phi,wm_tt_theta);
    auto wm_trt = sphDir(wm_trt_phi,wm_trt_theta);





    //auto wo_tt = getSmoothDir(wo,gammaI,gammaT,1);//
 //   auto wo_trt = getSmoothDir(wo,gammaI,gammaT,2);//

    auto [R1, cos_theta_t1, eta_it1, eta_ti1] = fresnel(dot(wo,wm_r), float(_eta));
    float whDotOut = dot(wo, wm_r);
    float  cosThetaT;
    auto eta_i = whDotOut>0?1/_eta:_eta;
    float  F = Fresnel::dielectricReflectance(1/_eta,whDotOut,cosThetaT);
    auto wh_r = (wi + wo).normalized();

    
    auto r_res = Vec3f (Fresnel::dielectricReflectance(1/_eta,wi.dot(wh_r)) * Microfacet::D(Microfacet::GGX,alpha_r,wm_local(wh_r,wm_r))
            * Microfacet::G(Microfacet::GGX,alpha_r,wm_local(wo,wm_r), wm_local(wi,wm_r),wh_r))/( 4 * abs(dot(wm_r,wo)));

    if(F==1)
        return r_res;
    auto wo_tt =  -refract(wo, wm_r, cos_theta_t1, eta_ti1);
    wo_tt = -((eta_i * whDotOut - (whDotOut>0?1:-1)*cosThetaT)* wm_r - eta_i* wo);
    auto wo_trt = -Reflect(wo_tt,wm_tt);



    auto wh_tt = normalize(wi + wo_tt * _eta);
    auto wh_trt = normalize(wi + wo_trt * _eta);

    if(dot(wm_tt,wo_tt)>0){
        int k  =1;
    }

   // return wi.y * wo.y>0?Spectrum(1,0,0):Spectrum (0,1,0)
    auto tt_res = Ntt;
   // return Spectrum(wo.y);
        tt_res   *= eval_dielectric(wi,wo_tt,wh_tt,wm_tt,alpha_tt);
            auto d  =dot(wm_trt,wo_trt);
            auto d1 =dot(wm_tt,wo_tt);
            auto d2 = dot(wm_tt,wi);
            auto d3= dot(wi,wm_trt);
//    if(d1<0)
//        throw("error");
    //return tt_res;

    //wo wm_r ,wo_tt ,wm_tt wo_trt ,wm_trt
    std::string phi_s;
    phi_s = std::to_string(phiO) +","+ std::to_string(wm_r_phi)+","+
    std::to_string(getPhi(wo_tt)) +","+ std::to_string(wm_tt_phi)+","+
    std::to_string(getPhi(wo_trt)) +","+ std::to_string(wm_trt_phi)+",";
    auto trt_res = Ntrt * eval_dielectric(wi,wo_trt,wh_trt,wm_trt,alpha_trt);
    //return trt_res;
    //return eval_dielectric(wi,wo_trt,wh_trt,wm_trt,alpha_trt);

//    if(dot(wm_r,wi)<0 || dot(wm_r,wo)<0 || dot(wm_r,wh_r)<0){
//        r_res = Spectrum(0);
//    }
//    auto tt_res = Ntt * eval_dielectric(wi,wo_tt,wh_tt,)
//
//    auto tt_res = Ntt * ggx.D(wm_local(wh_real,wh_tt),alpha_tt) * ggx.G(wm_local(getSmoothDir(wo,gammaI,gammaT,1),wh_r), wm_local(wi,wh_r),alpha_tt);
//    auto trt_res = Ntrt * ggx.D(wm_local(wh_real,wh_trt),alpha_trt) * ggx.G(wm_local(getSmoothDir(wo,gammaI,gammaT,2),wh_r), wm_local(wi,wh_r),alpha_trt);
 //   return trt_res;
    auto res =  trt_res;
    return res;
   // return tt_res;
  //  return Spectrum(0);
    res+=tt_res + r_res;
    return res;
    return res;
}

bool HairBcsdf::sample(SurfaceScatterEvent &event) const
{
    if (!event.requestedLobe.test(BsdfLobes::GlossyLobe))
        return false;

    Vec2f xiN = event.sampler->next2D();
    Vec2f xiM = event.sampler->next2D();

    float sinThetaI = event.wi.y();
    float cosThetaI = trigInverse(sinThetaI);
    float thetaI = std::asin(clamp(sinThetaI, -1.0f, 1.0f));

    float thetaIR   = thetaI - 2.0f*_scaleAngleRad;
    float thetaITT  = thetaI +      _scaleAngleRad;
    float thetaITRT = thetaI + 4.0f*_scaleAngleRad;

    // The following lines are just lobe selection
    float weightR   = _nR  ->weight(cosThetaI);
    float weightTT  = _nTT ->weight(cosThetaI);
    float weightTRT = _nTRT->weight(cosThetaI);

    const PrecomputedAzimuthalLobe *lobe;
    float v;
    float theta;

    float target = xiN.x()*(weightR + weightTT + weightTRT);
    if (target < weightR) {
        v = _vR;
        theta = thetaIR;
        lobe = _nR.get();
    } else if (target < weightR + weightTT) {
        v = _vTT;
        theta = thetaITT;
        lobe = _nTT.get();
    } else {
        v = _vTRT;
        theta = thetaITRT;
        lobe = _nTRT.get();
    }

    // Actual sampling of the direction starts here
    float sinThetaO = sampleM(v, std::sin(theta), std::cos(theta), xiM.x(), xiM.y());
    float cosThetaO = trigInverse(sinThetaO);

    float thetaO = std::asin(clamp(sinThetaO, -1.0f, 1.0f));
    float thetaD = (thetaO - thetaI)*0.5f;
    float cosThetaD = std::cos(thetaD);

    float phi, phiPdf;
    lobe->sample(cosThetaD, xiN.y(), phi, phiPdf);


//    float gammaI = std::asin(clamp(h, - 1.0f, 1.0f));
//    float gammaT = std::asin(clamp(h / iorPrime, - 1.0f, 1.0f));
//
//
//    deltaphi = Phi(gammaI, gammaT, p) + SampleTrimmedLogistic(u0[1], v, - PI, PI);
//    float phi = phiO + deltaphi;


    float sinPhi = std::sin(phi);
    float cosPhi = std::cos(phi);

    event.wo = Vec3f(sinPhi*cosThetaO, sinThetaO, cosPhi*cosThetaO);
    event.pdf = pdf(event);
    event.weight = eval(event)/event.pdf;
    event.sampledLobe = BsdfLobes::GlossyLobe;
    event.weight = (event.wi + 1.f)/2.f;
    return true;
}

float HairBcsdf::pdf(const SurfaceScatterEvent &event) const
{
    if (!event.requestedLobe.test(BsdfLobes::GlossyLobe))
        return 0.0f;

    float sinThetaI = event.wi.y();
    float sinThetaO = event.wo.y();
    float cosThetaI = trigInverse(sinThetaI);
    float cosThetaO = trigInverse(sinThetaO);
    float thetaI = std::asin(clamp(sinThetaI, -1.0f, 1.0f));
    float thetaO = std::asin(clamp(sinThetaO, -1.0f, 1.0f));
    float thetaD = (thetaO - thetaI)*0.5f;
    float cosThetaD = std::cos(thetaD);

    float phi = std::atan2(event.wo.x(), event.wo.z());
    if (phi < 0.0f)
        phi += TWO_PI;

    float thetaIR   = thetaI - 2.0f*_scaleAngleRad;
    float thetaITT  = thetaI +      _scaleAngleRad;
    float thetaITRT = thetaI + 4.0f*_scaleAngleRad;

    float weightR   = _nR  ->weight(cosThetaI);
    float weightTT  = _nTT ->weight(cosThetaI);
    float weightTRT = _nTRT->weight(cosThetaI);
    float weightSum = weightR + weightTT + weightTRT;

    float pdfR   = weightR  *M(_vR,   std::sin(thetaIR),   sinThetaO, std::cos(thetaIR),   cosThetaO);
    float pdfTT  = weightTT *M(_vTT,  std::sin(thetaITT),  sinThetaO, std::cos(thetaITT),  cosThetaO);
    float pdfTRT = weightTRT*M(_vTRT, std::sin(thetaITRT), sinThetaO, std::cos(thetaITRT), cosThetaO);

    return (1.0f/weightSum)*
          (pdfR  *  _nR->pdf(phi, cosThetaD)
         + pdfTT * _nTT->pdf(phi, cosThetaD)
         + pdfTRT*_nTRT->pdf(phi, cosThetaD));
}


void HairBcsdf::precomputeAzimuthalDistributions()
{
    const int Resolution = PrecomputedAzimuthalLobe::AzimuthalResolution;
    std::unique_ptr<Vec3f[]> valuesR  (new Vec3f[Resolution*Resolution]);
    std::unique_ptr<Vec3f[]> valuesTT (new Vec3f[Resolution*Resolution]);
    std::unique_ptr<Vec3f[]> valuesTRT(new Vec3f[Resolution*Resolution]);

    // Ideally we could simply make this a constexpr, but MSVC does not support that yet (boo!)
    #define NumPoints 140

    GaussLegendre<NumPoints> integrator;
    const auto points = integrator.points();
    const auto weights = integrator.weights();

    // Cache the gammaI across all integration points
    std::array<float, NumPoints> gammaIs;
    for (int i = 0; i < NumPoints; ++i)
        gammaIs[i] = std::asin(points[i]);

    // Precompute the Gaussian detector and sample it into three 1D tables.
    // This is the only part of the precomputation that is actually approximate.
    // 2048 samples are enough to support the lowest roughness that the BCSDF
    // can reliably simulate
    const int NumGaussianSamples = 2048;
    std::unique_ptr<float[]> Ds[3];
    for (int p = 0; p < 3; ++p) {
        Ds[p].reset(new float[NumGaussianSamples]);
        for (int i = 0; i < NumGaussianSamples; ++i)
            Ds[p][i] = D(_betaR, i/(NumGaussianSamples - 1.0f)*TWO_PI);
    }

    // Simple wrapped linear interpolation of the precomputed table
    auto approxD = [&](int p, float phi) {
        if(useGgx)
        return 1.f;
        float u = std::abs(phi*(INV_TWO_PI*(NumGaussianSamples - 1)));
        int x0 = int(u);
        int x1 = x0 + 1;
        u -= x0;
        return Ds[p][x0 % NumGaussianSamples]*(1.0f - u) + Ds[p][x1 % NumGaussianSamples]*u;
    };

    // Here follows the actual precomputation of the azimuthal scattering functions
    // The scattering functions are parametrized with the azimuthal angle phi,
    // and the cosine of the half angle, cos(thetaD).
    // This parametrization makes the azimuthal function relatively smooth and allows using
    // really low resolutions for the table (64x64 in this case) without any visual
    // deviation from ground truth, even at the lowest supported roughness setting
    for (int y = 0; y < Resolution; ++y) {
        float cosHalfAngle = y/(Resolution - 1.0f);

        // Precompute reflection Fresnel factor and reduced absorption coefficient
        float iorPrime = std::sqrt(Eta*Eta - (1.0f - cosHalfAngle*cosHalfAngle))/cosHalfAngle;
        float cosThetaT = std::sqrt(1.0f - (1.0f - cosHalfAngle*cosHalfAngle)*sqr(1.0f/Eta));
        Vec3f sigmaAPrime = _sigmaA/cosThetaT;

        // Precompute gammaT, f_t and internal absorption across all integration points
        std::array<float, NumPoints> fresnelTerms, gammaTs;
        std::array<Vec3f, NumPoints> absorptions;
        for (int i = 0; i < NumPoints; ++i) {
            gammaTs[i] = std::asin(clamp(points[i]/iorPrime, -1.0f, 1.0f));
            fresnelTerms[i] = Fresnel::dielectricReflectance(1.0f/Eta, cosHalfAngle*std::cos(gammaIs[i]));
            absorptions[i] = std::exp(-sigmaAPrime*2.0f*std::cos(gammaTs[i]));
        }

        for (int phiI = 0; phiI < Resolution; ++phiI) {
            float phi = TWO_PI*phiI/(Resolution - 1.0f);

            float integralR = 0.0f;
            Vec3f integralTT(0.0f);
            Vec3f integralTRT(0.0f);

            // Here follows the integration across the fiber width, h.
            // Since we were able to precompute most of the factors that
            // are constant w.r.t phi for a given h,
            // we don't have to do much work here.
            for (int i = 0; i < integrator.numSamples(); ++i) {
                float fR = fresnelTerms[i];
                Vec3f T = absorptions[i];

                float AR = fR;
                Vec3f ATT = (1.0f - fR)*(1.0f - fR)*T;
                Vec3f ATRT = ATT*fR*T;

                integralR   += weights[i]*approxD(0, phi - Phi(gammaIs[i], gammaTs[i], 0))*AR;
                integralTT  += weights[i]*approxD(1, phi - Phi(gammaIs[i], gammaTs[i], 1))*ATT;
                integralTRT += weights[i]*approxD(2, phi - Phi(gammaIs[i], gammaTs[i], 2))*ATRT;
            }

            valuesR  [phiI + y*Resolution] = Vec3f(0.5f*integralR);
            valuesTT [phiI + y*Resolution] = 0.5f*integralTT;
            valuesTRT[phiI + y*Resolution] = 0.5f*integralTRT;
        }
    }

    // Hand the values off to the helper class to construct sampling CDFs and so forth
    _nR  .reset(new PrecomputedAzimuthalLobe(std::move(valuesR)));
    _nTT .reset(new PrecomputedAzimuthalLobe(std::move(valuesTT)));
    _nTRT.reset(new PrecomputedAzimuthalLobe(std::move(valuesTRT)));
}

void HairBcsdf::prepareForRender()
{
    // Roughening/tightening of the different lobes as described in Marschner et al.'s paper.
    // Multiplied with Pi/2 to have similar range as the rough dielectric microfacet.
    // Clamped to some minimum roughness value to avoid oscillations in the azimuthal
    // scattering function.
    _betaR   = max(PI_HALF*_roughness, 0.04f);
    _betaTT  = _betaR*0.5f;
    _betaTRT = _betaR*2.0f;

    _vR   = _betaR  *_betaR;
    _vTT  = _betaTT *_betaTT;
    _vTRT = _betaTRT*_betaTRT;

    _scaleAngleRad = Angle::degToRad(_scaleAngleDeg);

    if (!_overridesSigmaA) {
        // The two melanin parameters allow easy reproduction of physical hair colors
        // based on the mixture of two pigments, eumelanin and pheomelanin, found in human hair
        // These RGB absorption values are taken from "An Energy-Conserving Hair Reflectance Model"
        const Vec3f eumelaninSigmaA = Vec3f(0.419f, 0.697f, 1.37f);
        const Vec3f pheomelaninSigmaA = Vec3f(0.187f, 0.4f, 1.05f);

        _sigmaA = _melaninConcentration*lerp(eumelaninSigmaA, pheomelaninSigmaA, _melaninRatio);
    }

    precomputeAzimuthalDistributions();
}


}
