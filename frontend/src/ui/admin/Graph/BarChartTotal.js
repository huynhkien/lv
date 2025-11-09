import React, { useEffect, useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import { apiGetTotal } from '../../../apis';

const BarChartExample = () => {
  const [count, setCount] = useState([]);
  const [selectedDay, setSelectedDay] = useState('');
  const [selectedMonth, setSelectedMonth] = useState(new Date().getMonth() + 1);
  const [selectedYear, setSelectedYear] = useState(new Date().getFullYear());
  const [viewType, setViewType] = useState('month'); // 'day', 'month', 'year'

  const fetchCount = async () => {
    let params = { year: selectedYear };
    
    if (viewType === 'month') {
        params.month = selectedMonth;
    } else if (viewType === 'day' && selectedDay) {
        params.month = selectedMonth;
        params.day = selectedDay;
    }
    
    const response = await apiGetTotal(params);
    if (response.success) {
        setCount(response.data);
    }
  };

  useEffect(() => {
    fetchCount();
  }, [selectedDay, selectedMonth, selectedYear, viewType]);

  const months = [
    { value: 1, label: 'Tháng 1' },
    { value: 2, label: 'Tháng 2' },
    { value: 3, label: 'Tháng 3' },
    { value: 4, label: 'Tháng 4' },
    { value: 5, label: 'Tháng 5' },
    { value: 6, label: 'Tháng 6' },
    { value: 7, label: 'Tháng 7' },
    { value: 8, label: 'Tháng 8' },
    { value: 9, label: 'Tháng 9' },
    { value: 10, label: 'Tháng 10' },
    { value: 11, label: 'Tháng 11' },
    { value: 12, label: 'Tháng 12' },
  ];

  const years = Array.from({ length: 10 }, (_, i) => new Date().getFullYear() - i);
  
  // Tạo danh sách ngày dựa trên tháng và năm được chọn
  const getDaysInMonth = (month, year) => {
    const daysCount = new Date(year, month, 0).getDate();
    return Array.from({ length: daysCount }, (_, i) => i + 1);
  };
  
  const days = getDaysInMonth(selectedMonth, selectedYear);

  // Kiểm tra có dữ liệu không
  const hasData = count && count.length > 0;

  // Format số tiền
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('vi-VN', {
      style: 'currency',
      currency: 'VND'
    }).format(value);
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border rounded shadow-lg">
          <p className="font-semibold">{payload[0].payload.date}</p>
          <p className="text-blue-600">
            Doanh thu: {formatCurrency(payload[0].value)}
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="w-full">
      <div className="mb-4 flex flex-wrap gap-4 items-center">
        <div className="flex gap-2">
          <button
            className={`px-4 py-2 rounded ${viewType === 'day' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            onClick={() => setViewType('day')}
          >
            Theo ngày
          </button>
          <button
            className={`px-4 py-2 rounded ${viewType === 'month' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            onClick={() => setViewType('month')}
          >
            Theo tháng
          </button>
          <button
            className={`px-4 py-2 rounded ${viewType === 'year' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
            onClick={() => setViewType('year')}
          >
            Theo năm
          </button>
        </div>

        {viewType === 'day' && (
          <select
            className="px-3 py-2 border rounded"
            value={selectedDay}
            onChange={(e) => setSelectedDay(e.target.value)}
          >
            <option value="">Tất cả các ngày</option>
            {days.map((day) => (
              <option key={day} value={day}>
                Ngày {day}
              </option>
            ))}
          </select>
        )}

        {(viewType === 'day' || viewType === 'month') && (
          <select
            className="px-3 py-2 border rounded"
            value={selectedMonth}
            onChange={(e) => setSelectedMonth(Number(e.target.value))}
          >
            {months.map((month) => (
              <option key={month.value} value={month.value}>
                {month.label}
              </option>
            ))}
          </select>
        )}

        <select
          className="px-3 py-2 border rounded"
          value={selectedYear}
          onChange={(e) => setSelectedYear(Number(e.target.value))}
        >
          {years.map((year) => (
            <option key={year} value={year}>
              Năm {year}
            </option>
          ))}
        </select>
      </div>

      {!hasData ? (
        <div className="flex items-center justify-center  border rounded ">
          <div className="text-center">
            <p className="text-gray-500 text-lg font-medium">Chưa có dữ liệu</p>
            <p className="text-gray-400 text-sm mt-1">
              Không có đơn hàng nào trong khoảng thời gian này
            </p>
          </div>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={count}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="date" 
              stroke="#8884d8"
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis 
              tickFormatter={(value) => `${(value / 1000000).toFixed(1)}M`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Bar 
              dataKey="totalAmount" 
              fill="#8884d8" 
              barSize={30}
              name="Doanh thu (VNĐ)"
            />
          </BarChart>
        </ResponsiveContainer>
      )}
      
      {hasData && (
        <div className="mt-4 p-4 bg-blue-50 rounded">
          <p className="font-semibold text-gray-700">
            Tổng doanh thu: {formatCurrency(count.reduce((sum, item) => sum + item.totalAmount, 0))}
          </p>
        </div>
      )}
    </div>
  );
};

export default BarChartExample;