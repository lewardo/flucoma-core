#pragma once

#include "data/FluidIndex.hpp"
#include "data/FluidTensor.hpp"
#include "data/TensorTypes.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>


// TODO:: REMOVE TEMPLATE THINGY AND TURN IT INTO CONSISTNE ALLOCATION OF MATRIX
namespace fluid {

template <typename idType, typename dataType, index N>
class FluidDataSeries
{

public:
  using FrameType = FluidTensor<dataType, N>;

  explicit FluidDataSeries() = default;
  ~FluidDataSeries() = default;

  // Construct from list of dimensions for each data point,
  // e.g. FluidDataSet(2, 3) is a dataset of 2x3 tensors
  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  FluidDataSeries(Dims... dims) 
  : mData(0, FluidTensor<dataType, N + 1>(0, dims...)), 
    mDim(dims...)
  {
    static_assert(sizeof...(dims) == N, "Number of dimensions doesn't match");
  }

  // Construct from existing tensors of ids and data points
  FluidDataSeries(FluidTensorView<const idType, 1>                    ids,
                  std::vector<FluidTensorView<const dataType, N + 1>> points)
      : mIds(ids), mData(points)
  {
    initFromData();
  }

  // Construct from existing tensors of ids and data points
  // (from convertible type for data, typically float -> double)
  template <typename U, typename T = dataType>
  FluidDataSeries(FluidTensorView<const idType, 1> ids,
                  std::vector<FluidTensorView<const U, N + 1>>  points,
                  std::enable_if_t<std::is_convertible<U, T>::value>* = nullptr)
      : mIds(ids), mData(points)
  {
    initFromData();
  }

  // Resize data point layout (if empty)
  template <typename... Dims,
            typename = std::enable_if_t<isIndexSequence<Dims...>()>>
  bool resize(Dims... dims)
  {
    static_assert(sizeof...(dims) == N, "Number of dimensions doesn't match");
    if (size() == 0)
    {
      mData = std::vector<FluidTensor<dataType, N + 1>>();
      mDim = FluidTensorSlice<N>(dims...);
      return true;
    }
    else
    {
      return false;
    }
  }

  template <typename U>
  bool addFrame(idType const& id, FluidTensorView<U, N> frame)
  {
    static_assert(std::is_convertible<U, dataType>::value,  "Can't convert between types");
    assert(sameExtents(mDim, frame.descriptor()));

    auto pos = mIndex.find(id);
    if (pos == mIndex.end())
    {
      FluidTensorView<U, N + 1> newPoint(frame);
      return addSeries(id, newPoint);
    } 

    mData[pos->second].resizeDim(0, 1);
    mData[pos->second].row(mData[pos->second].rows() - 1) <<= frame;

    return true;
  }

  template <typename U>
  bool addSeries(idType const& id, FluidTensorView<U, N + 1> series)
  {
    static_assert(std::is_convertible<U, dataType>::value,  "Can't convert between types");
    
    // dont crete another view 
    auto result = mIndex.insert({id, mData.size()});
    if (!result.second) return false;

    mData.emplace_back(series);

    mIds.resizeDim(0, 1);
    mIds(mIds.rows() - 1) = id;

    return true;
  }

  bool getFrame(idType const& id, index time, FluidTensorView<dataType, N> frame) const
  {
    auto pos = mIndex.find(id);
    if (pos == mIndex.end()) return false;

    if (time >= mData[pos->second].rows()) return false;

    frame <<= mData[pos->second].row(time);

    return true;
  }

  FluidTensorView<const dataType, N> getFrame(idType const& id, index time) const
  {
    auto pos = mIndex.find(id);
    if(pos != mIndex.end())
    {
      assert(time < mData[pos->second].rows());
      return mData[pos->second].row(time);
    }
    else { return FluidTensorView<const dataType, N>{nullptr, 0, 0}; }
  }

  bool getSeries(idType const& id, FluidTensorView<dataType, N + 1> series) const
  {
    auto pos = mIndex.find(id);
    if (pos == mIndex.end()) return false;

    series <<= mData[pos->second];

    return true;
  }

  FluidTensorView<const dataType, N + 1> getSeries(idType const& id) const
  {
    auto pos = mIndex.find(id);
    return pos != mIndex.end()
               ? mData[pos->second]
               : FluidTensorView<const dataType, N + 1>{nullptr, 0, 0, 0};
  }

  template <typename U>
  bool updateFrame(idType const& id, index time, FluidTensorView<U, N> frame)
  {
    static_assert(std::is_convertible<U, dataType>::value,  "Can't convert between types");
    
    auto pos = mIndex.find(id);
    if (pos == mIndex.end()) return false;

    if (time >= mData[pos->second].rows()) return false;
    mData[pos->second].row(time) <<= frame;

    return true;
  }

  template <typename U>
  bool updateSeries(idType const& id, FluidTensorView<U, N + 1> series)
  {
    static_assert(std::is_convertible<U, dataType>::value,  "Can't convert between types");

    auto pos = mIndex.find(id);
    if (pos == mIndex.end()) return false;
    else 
    {
      mData[pos->second].resizeDim(0, series.rows() - mData[pos->second].rows());
      mData[pos->second] <<= series;
    }

    return true;
  }

  bool removeFrame(idType const& id, index time)
  {
    auto pos = mIndex.find(id);
    if (pos == mIndex.end()) return false;

    index current = pos->second;
    if (time >= mData[current].rows()) return false;

    if(mData[current].rows() == 1) return removeSeries(id);
    else mData[current].deleteRow(time);

    return true;
  }

  bool removeSeries(idType const& id)
  {
    auto pos = mIndex.find(id);
    if (pos == mIndex.end()) return false;

    index current = pos->second;
    mData.erase(mData.begin() + current);
    mIds.deleteRow(current);
    mIndex.erase(id);

    for (auto& point : mIndex)
    {
      if (point.second > current) point.second--;
    }

    return true;
  }

  index getIndex(idType const& id) const
  {
    auto pos = mIndex.find(id);
    if (pos == mIndex.end()) return -1;
    else return pos->second;
  }

  index getNumFrames(idType const& id) const 
  {
    auto pos = mIndex.find(id);
    if (pos == mIndex.end()) return -1;
    else return mData[pos->second].rows();
  }

  std::vector<FluidTensorView<dataType, N + 1>> getData() 
  { 
    std::vector<FluidTensorView<dataType, N + 1>> viewVec(mData.size());

    // hacky fix to force conversion of vector of tensors to vector of views of mData
    // doesn't actually copy anything, it uses the FluidTensor ctor of FluidTensorView
    // which creates a view/ref, so ends up creating what we want
    std::copy(mData.begin(), mData.end(), std::back_inserter(viewVec));

    return viewVec; 
  }

  const std::vector<FluidTensorView<const dataType, N + 1>> getData() const 
  { 
    std::vector<FluidTensorView<const dataType, N + 1>> viewVec;

    // hacky fix to force conversion of vector to views of mData
    // doesn't actually copy anything, it uses the FluidTensor ctor of FluidTensorView
    // which creates a view/ref, so ends up creating what we want
    std::copy(mData.cbegin(), mData.cend(), std::back_inserter(viewVec));

    return viewVec; 
  }

  FluidTensorView<idType, 1>             getIds() { return mIds; }
  FluidTensorView<const idType, 1>       getIds() const { return mIds; }

  index pointSize() const { return mDim.size; }
  index dims() const { return mDim.size; }
  index size() const { return mIds.size(); }
  bool  initialized() const { return (size() > 0); }

  std::string printFrame(FluidTensorView<const dataType, N> frame,
                         index maxCols) const
  {
    using namespace std;
    ostringstream result;

    if (frame.size() < maxCols)
    {
      for (index c = 0; c < frame.size(); c++)
        result << setw(10) << setprecision(5) << frame(c);
    }
    else
    {
      for (index c = 0; c < maxCols / 2; c++) 
        result << setw(10) << setprecision(5) << frame(c);

      result << setw(10) << "...";

      for (index c = maxCols / 2; c > 0; c--)
        result << setw(10) << setprecision(5) << frame(frame.size() - c);
    }

    return result.str();
  }

  std::string printSeries(FluidTensorView<const dataType, N + 1> series,
                         index maxFrames, index maxCols) const
  {
    using namespace std;
    ostringstream result;

    if (series.rows() < maxFrames)
    {
      for (index t = 0; t < series.rows(); t++)
        result << setw(10) << "t" << t << ": " << printFrame(series.row(t), maxCols)
               << endl;
    }
    else
    {
      for (index t = 0; t < maxFrames / 2; t++)
        result << setw(10) << "t" << t << ": " << printFrame(series.row(t), maxCols)
               << endl;

      result << setw(10) << "..." << std::endl;

      for (index t = maxFrames / 2; t > 0; t--)
        result << setw(10) << "t" << (series.rows() - t) << ": " 
               << printFrame(series.row(size() - t), maxCols) << endl;
    }

    return result.str();
  }

  std::string print(index maxRows = 6, index maxFrames = 6, index maxCols = 6) const
  {
    using namespace std;
    ostringstream result;

    if (size() == 0) return "{}";
    result << endl << "points: " << size() << endl << "frame size: " << pointSize() << endl;

    if (size() < maxRows)
    {
      for (index r = 0; r < size(); r++)
        result << mIds(r) << ":" << endl << printSeries(mData[r], maxFrames, maxCols)
               << endl;
    }
    else
    {
      for (index r = 0; r < maxRows / 2; r++)
        result << mIds(r) << ":" << endl << printSeries(mData[r], maxFrames, maxCols)
               << endl;

      result << setw(10) << "⋮" << endl;

      for (index r = maxRows / 2; r > 0; r--)
        result << mIds(size() - r) << ":" << endl
               << printSeries(mData[size() - r], maxFrames, maxCols) << endl;
    }

    return result.str();
  }

private:
  void initFromData()
  {
    assert(mIds.rows() == mData.size());
    mDim = mData[0].cols();
    for (index i = 0; i < mIds.size(); i++) 
      mIndex.insert({mIds[i], i});
  }

  std::unordered_map<idType, index> mIndex;
  FluidTensor<idType, 1>            mIds;
  std::vector<FluidTensor<dataType, N + 1>> mData;
  FluidTensorSlice<N>               mDim; // dimensions for one frame
};
} // namespace fluid
