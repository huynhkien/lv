import React, { useEffect, useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from 'recharts';
import { apiGetCountRatings } from '../../../apis';

const BarChartExample = () => {
  const [count, setCount] = useState(null);
  const [selectedMonth, setSelectedMonth] = useState(new Date().getMonth() + 1);
  const [selectedYear, setSelectedYear] = useState(new Date().getFullYear());
  const [viewType, setViewType] = useState('month');

  const fetchCount = async () => {
    const params = viewType === 'month' 
      ? { month: selectedMonth, year: selectedYear }
      : { year: selectedYear };
      
    const response = await apiGetCountRatings(params);
    if (response.success) {
      setCount(response.data);
    }
  };

  useEffect(() => {
    fetchCount();
  }, [selectedMonth, selectedYear, viewType]);

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

  // Kiểm tra xem có dữ liệu không
  const hasData = count && Object.values(count).some(value => value > 0);

  return (
    <div className="w-full">
      <div className="mb-4 flex gap-4 items-center">

        {viewType === 'month' && (
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
        <div className="flex items-center justify-center border rounded">
          <div className="text-center">
            <p>Chưa có dữ liệu</p>
            <p>
              Không có đánh giá nào trong {viewType === 'month' ? `tháng ${selectedMonth}` : 'năm'} {selectedYear}
            </p>
          </div>
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={count ? Object.entries(count).map(([key, value]) => ({ 
            name: key, 
            'Số lượng': value 
          })) : []}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="Số lượng" fill="#8884d8" barSize={50} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};

export default BarChartExample;