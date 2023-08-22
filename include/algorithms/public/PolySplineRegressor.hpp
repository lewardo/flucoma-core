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
    using FluidVector = FluidTensor<index, 1>;

public:
    explicit PolySplineRegressor() = default;
    ~PolySplineRegressor() = default;

    template<typename = std::enable_if_t<S == PolySplineType::Polynomial>>
    void init(index degree, index dims, double penalty = 0.0)
    {
        mInitialized = true;
        setDegree(degree);
        setDims(dims);
        setPenalty(penalty);
    };

    template<typename = std::enable_if_t<S == PolySplineType::Spline>>
    void init(index degree, index dims, index knots, double penalty = 0.0)
    {
        mInitialized = true;
        setDegree(degree);
        setDims(dims);
        setKnots(knots);
        setPenalty(penalty);
    };

    index degree()      const { return mInitialized ? asSigned(mDegree) : 0; };
    index numCoeffs()   const { return mInitialized ? asSigned(mDegree + mKnots + 1) : 0; }
    index numKnots()    const { return mInitialized ? asSigned(mKnots) : 0; }
    index dims()        const { return mInitialized ? asSigned(mDims) : 0; };
    index size()        const { return mInitialized ? asSigned(mDegree) : 0; };

    double penalty()    const { return mInitialized ? mFilterFactor : 0.0; };

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

    void setPenalty(double penalty) 
    {
        if (mFilterFactor == penalty) return;

        mFilterFactor = penalty;
        mRegressed = false;
    }

    void setKnots(index knots)
    {
        if (mKnots == knots) return;

        mKnots = knots;
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
            generateKnotQuantiles(input.col(i));
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
            generateKnotQuantiles(in.col(i));
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
    template <PolySplineType T = S, std::enable_if_t<T == PolySplineType::Polynomial, int> = 0>
    void generateFilterMatrix() const
    {
        mFilterMatrix = mFilterFactor * MatrixXd::Identity(numCoeffs(), numCoeffs());
    }

    template <PolySplineType T = S, std::enable_if_t<T == PolySplineType::Spline, int> = 0>
    void generateFilterMatrix() const
    {
        mFilterMatrix = MatrixXd::Zero(numCoeffs(), numCoeffs());
        mFilterMatrix.bottomRightCorner(numKnots(), numKnots()) = mFilterFactor * MatrixXd::Identity(numKnots(), numKnots());
    }

    // naive splitting of the (min, max) range, prone to statistical anomalies so filtering of input values could be done here
    void generateKnotQuantiles(Eigen::Ref<VectorXd> in) const
    {
        index min = in.minCoeff(), max = in.maxCoeff();
        index range = max - min;
        double stride = range / (mKnots + 1);

        for (index i = 1; i < mKnots + 1; ++i)
        {
            mKnotQuantiles.push_back(min + i * stride);
        }
    };


    index mDegree       {2};
    index mDims         {1};
    index mKnots        {(index) S * 4};

    bool  mRegressed    {false};
    bool  mInitialized  {false};

    double mFilterFactor {0};

    MatrixXd    mCoefficients;

    mutable std::vector<index> mKnotQuantiles;
    mutable MatrixXd mDesignMatrix;
    mutable MatrixXd mFilterMatrix;
};

using PolynomialRegressor = PolySplineRegressor<PolySplineType::Polynomial>;
using SplineRegressor     = PolySplineRegressor<PolySplineType::Spline>;

} // namespace algorithm
} // namespace fluid