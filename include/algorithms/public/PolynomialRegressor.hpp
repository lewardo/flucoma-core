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

class PolynomialRegressor
{
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using ArrayXd  = Eigen::ArrayXd;

public:
    explicit PolynomialRegressor() = default;
    ~PolynomialRegressor() = default;

    void init(index degree, index dims, double tikhonov = 0.0)
    {
        mInitialized = true;
        setDegree(degree);
        setDims(dims);
        setTikhonov(tikhonov);
    };

    index degree()      const { return mInitialized ? asSigned(mDegree) : 0; };
    double tihkonov()   const { return mInitialized ? mTikhonovFactor : 0.0; };
    index dims()        const { return mInitialized ? asSigned(mDims) : 0; };
    index size()        const { return mInitialized ? asSigned(mDegree) : 0; };

    void clear() { mRegressed = false; }

    bool    regressed()     const { return mRegressed; };
    bool    initialized()   const { return mInitialized; };

    void setDegree(index degree) {
        if (mDegree == degree) return;

        mDegree = degree;
        mCoefficients.conservativeResize(numCoeffs(), mDims);
        mRegressed = false;
    }

    void setDims(index dims) {
        if (mDims == dims) return;

        mDims = dims;
        mCoefficients.conservativeResize(numCoeffs(), mDims);
        mRegressed = false;
    }

    void setTikhonov(double tikhonov) {
        if (mTikhonovFactor == tikhonov) return;

        mTikhonovFactor = tikhonov;
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

        generateTikhonovFilter(mDegree + 1);

        for(index i = 0; i < mDims; ++i)
        {
            generateDesignMatrix(input.col(i));
            
            // tikhonov/ridge regularisation, given Ax = y where x could be noisy
            // optimise the value _x = (A^T . A + R^T . R)^-1 . A^T . y
            // where R is a tikhonov filter matrix, in case of ridge regression of the form a.I
            Eigen::MatrixXd transposeDesignTikhonovProduct = mDesignMatrix.transpose() * mDesignMatrix + mTikhonovMatrix.transpose() * mTikhonovMatrix;
            mCoefficients.col(i) = transposeDesignTikhonovProduct.inverse() * mDesignMatrix.transpose() * output.col(i);
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
        VectorXd designColumn = VectorXd::Ones(in.size());
        Eigen::ArrayXd inArray = in.array();

        mDesignMatrix.conservativeResize(in.size(), numCoeffs());

        for (index i = 0; i < mDegree + 1; ++i, designColumn = designColumn.array() * inArray) 
            mDesignMatrix.col(i) = designColumn;
        
        if (isSpline())
        {
            designColumn = inArray;

            for (index k = 0; k < knots.size(); ++k) 
        }
    }

    void generateFilterMatrix() {
        if (isPoly()) generateTikhonovFilter(numCoeffs());
        if (isSpline()) generatePenalisationFilter(mDegree + 1, mKnots);
    }

    // currently only ridge normalisation with scaled identity matrix as tikhonov filter for polynomial
    void generateTikhonovFilter(index size)
    {
        mFilterMatrix = mTikhonovFactor * MatrixXd::Identity(size, size);
    };

    void generatePenalisationFilter(index mask, index size)
    {
        mFilterMatrix = MatrixXd::Zero(mask + size, mask + size);
        mFilterMatrix.bottomRightCorner(size, size) = MatrixXd::Identity(size, size);
    }

    index mDegree       {2};
    index mDims         {1};
    index mKnots        {0};
    bool  mRegressed    {false};
    bool  mInitialized  {false};

    double mTikhonovFactor {0};

    MatrixXd mCoefficients;
    VectorXd knots;

    mutable MatrixXd mDesignMatrix;
    mutable MatrixXd mFilterMatrix;
};

} // namespace algorithm
} // namespace fluid