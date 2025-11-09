import React from 'react';
import {LineChart, BarChart, PieChart, BarChartTotal } from "../../Index";

const Graph = () => {
  return (
    <div>
    <div className="header">
      <div className="left">
        <h1>Thống kê</h1>
      </div>
    </div>
    <div className="bottom-data">
      <div className="orders">
        <h5 className="text-center">Trạng thái đơn hàng</h5>
       <PieChart/>
      </div>
      <div className="orders">
      <h5 className="text-center">Đánh giá sản phẩm</h5>
      <BarChart/>
      </div>
    </div>
    <div className="bottom-data">
    <div className="orders">
    <h5 className="text-center">Doanh thu</h5>
        <BarChartTotal/>
     </div>
    </div>
  </div>
  )
}

export default Graph