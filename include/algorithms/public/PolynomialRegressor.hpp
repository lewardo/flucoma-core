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
public:
    explicit PolynomialRegressor() = default;
    ~PolynomialRegressor() = default;

    void init(index degree, index dim)
    {
        mInitialized = true;
        mDegree = degree;
        mSize = dim;
    };

    index dims() const { return asSigned(mDegree); };
    index size() const { return asSigned(mSize); };

    void clear() { mRegressed = false; }

    bool    regressed()     const { return mRegressed; };
    bool    initialized()   const { return mInitialized; };

    void setDegree(index degree) {
        if (mDegree == degree) return;

        mDegree = degree;
        mCoefficients.conservativeResize(mDegree + 1, mSize);
        mRegressed = false;
    }

    void setSize(index dim) {
        if (mSize == dim) return;

        mSize = dim;
        mCoefficients.conservativeResize(mDegree + 1, mSize);
        mRegressed = false;
    }

    void calculateRegressionCoefficients(InputRealMatrixView in, 
                                         InputRealMatrixView out,
                                         Allocator& alloc = FluidDefaultAllocator())
    {
        using namespace _impl;

        ScopedEigenMap<Eigen::MatrixXd> input(in.rows(), in.cols(), alloc), 
          output(out.rows(), out.cols(), alloc);
        input = asEigen<Eigen::Array>(in);
        output = asEigen<Eigen::Array>(out);

        for(index i = 0; i < mSize; ++i)
        {
            generateDesignMatrix(input.col(i));

            Eigen::MatrixXd transposeProduct = mDesignMatrix.transpose() * mDesignMatrix;
            mCoefficients.col(i) = transposeProduct.inverse() * mDesignMatrix.transpose() * output.col(i);
        }
        

        mRegressed = true;
    };

    void getCoefficients(RealMatrixView coefficients) const
    {
       _impl::asEigen<Eigen::Array>(coefficients) = mCoefficients;   
    };

    void setCoefficients(InputRealMatrixView coefficients)
    {
        setDegree(coefficients.rows() - 1);
        setSize(coefficients.cols());

        mCoefficients = _impl::asEigen<Eigen::Array>(coefficients);
        mRegressed = true;
    }

    void getMappedSpace(InputRealMatrixView in, 
                        RealMatrixView out, 
                        Allocator& alloc = FluidDefaultAllocator()) const
    {
        using namespace _impl;

        ScopedEigenMap<Eigen::MatrixXd> input(in.rows(), in.cols(), alloc),
          output(out.rows(), out.cols(), alloc);
        input = asEigen<Eigen::Array>(in);
        output = asEigen<Eigen::Array>(out);

        calculateMappings(input, output);

        asEigen<Eigen::Array>(out) = output;
    }

private:
    void calculateMappings(Eigen::Ref<Eigen::MatrixXd> in, Eigen::Ref<Eigen::MatrixXd> out) const
    {
        for(index i = 0; i < mSize; ++i)
        {
            generateDesignMatrix(in.col(i));
            out.col(i) = mDesignMatrix * mCoefficients.col(i);
        }
    }

    void generateDesignMatrix(Eigen::Ref<Eigen::VectorXd> in) const
    {
        Eigen::VectorXd designColumn = Eigen::VectorXd::Ones(in.size());
        Eigen::ArrayXd inArray = in.array();

        mDesignMatrix.conservativeResize(in.size(), mDegree + 1);

        for(index i = 0; i < mDegree + 1; ++i, designColumn = designColumn.array() * inArray) 
            mDesignMatrix.col(i) = designColumn;
    }

    index mDegree       {2};
    index mSize          {2};
    bool  mRegressed    {false};
    bool  mInitialized  {false};

    mutable Eigen::MatrixXd mDesignMatrix;
    Eigen::MatrixXd mCoefficients;

};

} // namespace algorithm
} // namespace fluid