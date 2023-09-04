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

#include "../util/FluidEigenMappings.hpp"
#include "../../data/FluidDataSet.hpp"
#include "../../data/FluidIndex.hpp"
#include "../../data/FluidMemory.hpp"
#include "../../data/FluidTensor.hpp"
#include "../../data/TensorTypes.hpp"
#include <Eigen/Core>
#include <random>

namespace fluid {
namespace algorithm {

class MatrixParam : public RealMatrix
{
  using EigenMatrixMap = Eigen::Map<Eigen::MatrixXd>;
  using EigenArrayXXMap = Eigen::Map<Eigen::ArrayXXd>;

public:
  template <typename... Args>
  MatrixParam(Args&&... args)
      : RealMatrix(std::forward<Args>(args)...),
        mMatrix{this->data(), this->rows(), this->cols()},
        mArray{this->data(), this->rows(), this->cols()}
  {}

  EigenMatrixMap  matrix() { return mMatrix; }
  EigenArrayXXMap array() { return mArray; }

private:
  EigenMatrixMap  mMatrix;
  EigenArrayXXMap mArray;
};

class VectorParam : public RealVector
{
  using EigenVectorMap = Eigen::Map<Eigen::VectorXd>;
  using EigenArrayXMap = Eigen::Map<Eigen::ArrayXd>;

public:
  template <typename... Args>
  VectorParam(Args&&... args)
      : RealVector(std::forward<Args>(args)...),
        mMatrix{this->data(), this->size()}, mArray{this->data(), this->size()}
  {}

  EigenVectorMap matrix() { return mMatrix; }
  EigenArrayXMap array() { return mArray; }

private:
  EigenVectorMap mMatrix;
  EigenArrayXMap mArray;
};


template <class CellType>
class Recur
{
  using CellSeries = rt::vector<CellType>;
  using CellState = typename CellType::StateType;
  using ParamType = typename CellType::ParamType;
  using ParamPtr = typename CellType::ParamPtr;
  using ParamLock = typename CellType::ParamLock;

public:
  explicit Recur() = default;
  ~Recur() = default;

  index size() const { return mInitialized ? asSigned(mNodes.size()) : 0; }
  index initialized() const { return mInitialized; }
  index dims() const { return mInitialized ? mOutSize : 0; }

  bool trained() const { return mInitialized ? mTrained : false; }
  void setTrained() const
  {
    if (mInitialized) mTrained = true;
  }

  ParamPtr getParams() const { return mParams; }

  void clear()
  {
    mParams.reset();
    mNodes.clear();
  };

  void init(index inSize, index outSize)
  {
    mParams = std::make_shared<ParamType>(inSize, outSize);

    mInSize = inSize;
    mOutSize = outSize;

    mInitialized = true;
  };

  double fit(InputRealMatrixView input, InputRealMatrixView output)
  {
    // check the input sizes and check either N-N or N-1
    assert(input.cols() == mInSize);
    assert(output.cols() == mOutSize);
    assert(input.rows() == output.rows() || output.rows() == 1);

    RealVector nowState(mParams->mOutSize), nowHidden(mParams->mOutSize),
        nextState(mParams->mOutSize), nextHidden(mParams->mOutSize);

    for (index i = 0; i < input.rows(); ++i)
    {
      mNodes.emplace_back(mParams);
      mNodes.back().forwardFrame(input.row(i), nowState, nowHidden, nextState,
                                 nextHidden);

      nowState << nextState;
      nowHidden << nextHidden;
    }


    for (index row = input.rows() - 1; row > 0; --row)
    {
      mNodes.back().forwardFrame(input.row(i), nowState, nowHidden, nextState,
                                 nextHidden);

      nowState << nextState;
      nowHidden << nextHidden;
    }
  };

  void process(InputRealMatrixView input, RealMatrixView output)
  {
    assert(input.cols() == mInSize);
    assert(output.cols() == mOutSize);
    assert(input.rows() == output.rows() || output.rows() == 1);

    Cell       lstm(mParams);
    RealVector nowState(mParams->mOutSize), nowHidden(mParams->mOutSize),
        nextState(mParams->mOutSize), nextHidden(mParams->mOutSize);

    nowState.fill(0.0);
    nowHidden.fill(0.0);

    for (index i = 0; i < input.rows(); ++i)
    {
      lstm.forwardFrame(input.row(i), nowState, nowHidden, nextState,
                        nextHidden);

      nowState << nextState;
      nowHidden << nextHidden;

      if (output.rows() > 1 || row == input.rows() - 1)
      {
        output.row(i) <<= nowHidden;
      }
    }
  };

private:
  bool mInitialized{false};
  bool mTrained{false};

  index mInSize, mOutSize;

  // rt vector of cells (each have ptr to params)
  // pointer rather than ref so Recur can be default constructible
  CellSeries mNodes;
  ParamLock  mParams;
};

} // namespace algorithm
} // namespace fluid