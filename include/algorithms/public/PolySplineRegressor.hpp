/*
Part of the Fluid Corpus Manipulation Project (http://www.flucoma.org/)
Copyright University of Huddersfield.
Licensed under the BSD-3 License.
See license.md file in the project root for full license information.
This project has received funding from the European Research Council (ERC)
under the European Union’s Horizon 2020 research and innovation programme
(grant agreement No 725899).
*/

#pragma once

#include "../util/AlgorithmUtils.hpp"
#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <cmath>

namespace fluid {
namespace algorithm {

enum class PolySplineType
{
    Polynomial = 0,
    Spline
};

template <PolySplineType S>
class PolySplineRegressor
{
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using ArrayXd  = Eigen::ArrayXd;

public:
    explicit PolySplineRegressor() = default;
    ~PolySplineRegressor() = default;

    template<typename = std::enable_if_t<S == PolySplineType::Polynomial>>
    void init(index degree, index dims, double tikhonov = 0.0)
    {
        mInitialized = true;
        setDegree(degree);
        setDims(dims);
        setTikhonov(tikhonov);
    };

    template<typename = std::enable_if_t<S == PolySplineType::Spline>>
    void init(index degree, index dims, index knots, VectorXd knotQuantiles)
    {
        mInitialized = true;
        setDegree(degree);
        setDims(dims);
        setKnots(knots, knotQuantiles);
    };

    index degree()      const { return mInitialized ? asSigned(mDegree) : 0; };
    index numCoeffs()   const { return mInitialized ? asSigned(mDegree + mKnots + 1) : 0; }
    index numKnots()    const { return mInitialized ? asSigned(mKnots) : 0; }
    index dims()        const { return mInitialized ? asSigned(mDims) : 0; };
    index size()        const { return mInitialized ? asSigned(mDegree) : 0; };

    double tihkonov()   const { return mInitialized ? mTikhonovFactor : 0.0; };

    void clear() { mRegressed = false; }

    constexpr bool isPoly()   const { return S == PolySplineType::Polynomial; }
    constexpr bool isSpline() const { return S == PolySplineType::Spline; }

    bool    regressed()     const { return mRegressed; };
    bool    initialized()   const { return mInitialized; };

    void setDegree(index degree) 
    {
        if (mDegree == degree) return;

        mDegree = degree;
        mCoefficients.conservativeResize(numCoeffs(), mDims);
        mRegressed = false;
    }

    void setDims(index dims) 
    {
        if (mDims == dims) return;

        mDims = dims;
        mCoefficients.conservativeResize(numCoeffs(), mDims);
        mRegressed = false;
    }

    void setTikhonov(double tikhonov) 
    {
        if (mTikhonovFactor == tikhonov) return;

        mTikhonovFactor = tikhonov;
        mRegressed = false;
    }

    void setKnots(index knots, ArrayXidx knotQuantiles) 
    {
        if (mKnots.isApprox(knotQuantiles)) return;
        if (knotQuantiles.size() != knots 
            && (knotQuantiles.size() > 1 
            ||  knotQuantiles[0] != -1)) return;

        mKnots = knots;
        mKnotQuantiles = knotQuantiles;
        mRegressed = false;
    }

    void regress(InputRealMatrixView in, 
                 InputRealMatrixView out,
                 Allocator& alloc = FluidDefaultAllocator())
    {
        using namespace _impl;

        ScopedEigenMap<MatrixXd> input(in.rows(), in.cols(), alloc), 
          output(out.rows(), out.cols(), alloc);
        input = asEigen<Eigen::Array>(in);
        output = asEigen<Eigen::Array>(out);

        generateFilterMatrix();

        for(index i = 0; i < mDims; ++i)
        {
            generateDesignMatrix(input.col(i));
            
            MatrixXd transposeDesignFilterProduct = mDesignMatrix.transpose() * mDesignMatrix + mFilterMatrix.transpose() * mFilterMatrix;
            mCoefficients.col(i) = transposeDesignFilterProduct.inverse() * mDesignMatrix.transpose() * output.col(i);
        }
        

        mRegressed = true;
    };

    void getCoefficients(RealMatrixView coefficients) const
    {
       if (mInitialized) _impl::asEigen<Eigen::Array>(coefficients) = mCoefficients;   
    };

    void setCoefficients(InputRealMatrixView coefficients)
    {
        if(!mInitialized) mInitialized = true;
        
        setDegree(coefficients.rows() - 1);
        setDims(coefficients.cols());

        mCoefficients = _impl::asEigen<Eigen::Array>(coefficients);
        mRegressed = true;
    }

    void process(InputRealMatrixView in, 
                 RealMatrixView out, 
                 Allocator& alloc = FluidDefaultAllocator()) const
    {
        using namespace _impl;

        ScopedEigenMap<MatrixXd> input(in.rows(), in.cols(), alloc),
          output(out.rows(), out.cols(), alloc);
        input = asEigen<Eigen::Array>(in);
        output = asEigen<Eigen::Array>(out);

        calculateMappings(input, output);

        asEigen<Eigen::Array>(out) = output;
    }

private:
    void calculateMappings(Eigen::Ref<MatrixXd> in, Eigen::Ref<MatrixXd> out) const
    {
        for(index i = 0; i < mDims; ++i)
        {
            generateDesignMatrix(in.col(i));
            out.col(i) = mDesignMatrix * mCoefficients.col(i);
        }
    }

    void generateDesignMatrix(Eigen::Ref<VectorXd> in) const
    {
        ArrayXd designColumn = VectorXd::Ones(in.size()),
                inArray = in.array();

        mDesignMatrix.conservativeResize(in.size(), numCoeffs());

        for (index i = 0; i < mDegree + 1; ++i)
        {
            mDesignMatrix.col(i) = designColumn;
            designColumn = designColumn * inArray;
        }
        
        if (isSpline())
        {
            for (index k = mDegree + 1; k < numCoeffs(); ++k)
            {
                designColumn = inArray - mKnotQuantiles[k];
                designColumn = designColumn.max(ArrayXd::Zero(in.size()));
                designColumn = designColumn.pow(mDegree);
                mDesignMatrix.col(k) = designColumn;
            }
        }
    }

    // currently only ridge normalisation with scaled identity matrix as tikhonov filter for polynomial
    template <typename = std::enable_if_t<S == PolySplineType::Polynomial>>
    void generateFilterMatrix() const
    {
        mFilterMatrix = mTikhonovFactor * MatrixXd::Identity(numCoeffs(), numCoeffs());
    }

    template <typename = std::enable_if_t<S == PolySplineType::Spline>>
    void generateFilterMatrix() const
    {
        mFilterMatrix = MatrixXd::Zero(numCoeffs(), numCoeffs());
        mFilterMatrix.bottomRightCorner(numKnots(), numKnots()) = MatrixXd::Identity(numKnots(), numKnots());
    }

    index mDegree       {2};
    index mDims         {1};
    index mKnots        {(index)S * 4};

    bool  mRegressed    {false};
    bool  mInitialized  {false};

    double mTikhonovFactor {0};

    MatrixXd  mCoefficients;
    ArrayXidx mKnotQuantiles;

    mutable MatrixXd mDesignMatrix;
    mutable MatrixXd mFilterMatrix;
};

using PolynomialRegressor = PolySplineRegressor<PolySplineType::Polynomial>;
using SplineRegressor     = PolySplineRegressor<PolySplineType::Spline>;

} // namespace algorithm
} // namespace fluid