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
    using MatrixXi = Eigen::Matrix<index, -1, -1>;
    using VectorXd = Eigen::VectorXd;
    using VectorXi = Eigen::Matrix<index, -1, 1>;
    using ArrayXd  = Eigen::ArrayXd;

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
        setNumKnots(knots);
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
        mCoefficients.conservativeResize(numCoeffs(), dims());
        mRegressed = false;
    }

    void setDims(index dim) 
    {
        if (mDims == dim) return;

        mDims = dim;
        mCoefficients.conservativeResize(numCoeffs(), dims());
        mRegressed = false;
    }

    void setPenalty(double penalty) 
    {
        if (mFilterFactor == penalty) return;

        if constexpr (S == PolySplineType::Spline) mFilterFactor = std::sqrt(penalty); // to compensate for the filter matrix transpose product
        else mFilterFactor = penalty;

        mRegressed = false;
    }

    template<typename = std::enable_if_t<S == PolySplineType::Spline>>
    void setNumKnots(index knots)
    {
        if (mKnots == knots) return;

        mKnots = knots;
        mKnotQuantiles.conservativeResize(numKnots(), dims());
        mRegressed = false;
    }

    template<typename = std::enable_if_t<S == PolySplineType::Spline>>
    void getKnots(RealMatrixView knots)
    {
       if (mInitialized) _impl::asEigen<Eigen::Array>(knots) = mKnotQuantiles;   
    }

    template<typename = std::enable_if_t<S == PolySplineType::Spline>>
    void setKnots(InputRealMatrixView knots)
    {
        if(!mInitialized) mInitialized = true;

        setNumKnots(knots.rows());
        mKnotQuantiles = _impl::asEigen<Eigen::Array>(knots);
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

        for(index i = 0; i < dims(); ++i)
        {            
            if constexpr (S == PolySplineType::Spline) 
            {
                generateKnotQuantiles(input.col(i), mKnotQuantiles.col(i));
                generateDesignMatrix(input.col(i), mKnotQuantiles.col(i));
            }
            else generateDesignMatrix(input.col(i));

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
    void calculateMappings(const Eigen::Ref<const MatrixXd>& in, Eigen::Ref<MatrixXd> out) const
    {
        for(index i = 0; i < mDims; ++i)
        {
            if constexpr (S == PolySplineType::Spline) generateDesignMatrix(in.col(i), mKnotQuantiles.col(i));
            else generateDesignMatrix(in.col(i));

            out.col(i) = mDesignMatrix * mCoefficients.col(i);
        }
    }

    void generateDesignMatrix(const Eigen::Ref<const VectorXd>& in) const
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
    void generateDesignMatrix(const Eigen::Ref<const VectorXd>& in, const Eigen::Ref<const VectorXi>& quantiles) const
    {
        mFilterMatrix = mFilterFactor * MatrixXd::Identity(numCoeffs(), numCoeffs());
    }

    void generateFilterMatrix() const
    {
        if constexpr (S == PolySplineType::Spline)
        {
            mFilterMatrix = MatrixXd::Zero(numCoeffs(), numCoeffs());
            mFilterMatrix.bottomRightCorner(numKnots(), numKnots()) = mFilterFactor * MatrixXd::Identity(numKnots(), numKnots());
        } 
        else // currently only ridge normalisation with scaled identity matrix as tikhonov filter for polynomial
            mFilterMatrix = mFilterFactor * MatrixXd::Identity(numCoeffs(), numCoeffs());
    }

    // naive splitting of the (min, max) range, prone to statistical anomalies so filtering of input values could be done here
    void generateKnotQuantiles(const Eigen::Ref<const VectorXd>& in, Eigen::Ref<VectorXi> quantiles)
    {
        double min = in.minCoeff(), max = in.maxCoeff();
        double range = max - min;
        double stride = range / (mKnots + 1);

        for (index i = 1; i < mKnots + 1; ++i) quantiles[i - 1] = static_cast<index>(min + i * stride);
    }


    index mDegree       {2};
    index mDims         {1};
    index mKnots        {(index) S * 4};

    bool  mRegressed    {false};
    bool  mInitialized  {false};

    double mFilterFactor {0};

    MatrixXd mCoefficients;
    MatrixXi mKnotQuantiles;

    mutable MatrixXd mDesignMatrix;
    mutable MatrixXd mFilterMatrix;
};

using PolynomialRegressor = PolySplineRegressor<PolySplineType::Polynomial>;
using SplineRegressor     = PolySplineRegressor<PolySplineType::Spline>;

} // namespace algorithm
} // namespace fluid