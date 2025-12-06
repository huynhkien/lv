import React from 'react'
import { useEffect, useState } from 'react';
import { apiDeleteUser, apiGetAllOrder, apiGetUsers} from '../../../apis';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';
import { InputText } from 'primereact/inputtext';
import { FaEdit, FaTrash } from 'react-icons/fa';
import { toast } from 'react-toastify';

const Page = () => {
  const [users, setUsers] = useState([]);
  const [userPotentials, setUserPotentials] = useState([]);
  const [orders, setOrders] = useState([]);
  const [loading, setLoading] = useState(true);
  const [globalFilter, setGlobalFilter] = useState('');
  const [isSmallScreen, setIsSmallScreen] = useState(false);

  useEffect(() => {
    const handleResize = () => setIsSmallScreen(window.innerWidth < 700);
    window.addEventListener('resize', handleResize);
    handleResize();

    return () => window.removeEventListener('resize', handleResize);
  }, []);
  const fetchOrders = async() => {
    const response = await apiGetAllOrder();
    if(response.success){
        setOrders(response?.data);
    }
  }
  const fetchUsers = async () => {
    const response = await apiGetUsers();
    if (response.success) setUsers(response.userData);
    setLoading(false);
  };
  

  useEffect(() => {
    fetchOrders();
    fetchUsers();
  }, []);
  const findCustomersWithMoreThanOrders = () => {
  // Đếm số đơn hàng của mỗi user
  const orderCountByUser = {};
  
  orders.forEach(item => {
    const userId = item.user?._id;
    if (userId) {
      orderCountByUser[userId] = (orderCountByUser[userId] || 0) + 1;
    }
  });
  
  // Lọc ra các user cóđơn hàng
  const customersWithMoreThanOrders = users.filter(user => {
    return orderCountByUser[user._id] > 1;
  }).map(user => ({
    ...user,
    orderCount: orderCountByUser[user._id]
  }));
  
  return customersWithMoreThanOrders;
};

// Sử dụng trong useEffect hoặc khi cần
useEffect(() => {
  if (orders.length > 0 && users.length > 0) {
    const result = findCustomersWithMoreThanOrders();
    if(result){
        setUserPotentials(result);
    }
  }
}, [orders, users]);
  const filterRole = userPotentials
  ?.filter(el => el?.role === '2004')
  ?.filter(item => {
    const searchText = globalFilter.toLowerCase();
    return (
      item.name.toLowerCase().includes(searchText) ||
      item.email.toLowerCase().includes(searchText) ||
      item.phone.toLowerCase().includes(searchText) ||
      item.address.toLowerCase().includes(searchText) ||
      (item.role === '2004' && 'Khách hàng'.includes(searchText)) 
    );
  });


  const roleBodyTemplate = (rowData) => {
    return rowData.role === '2004' ? 'Khách hàng' : 'Vô danh';
  };
  const header = (
    <div className='p-inputgroup flex-1 my-2'>
        <InputText type='text' placeholder='Tìm kiếm' className='p-inputtext p-component p-2' onChange={(e) => setGlobalFilter(e.target.value)} />
    </div>
  );

  return (
    <div>
      <div className='header'>
        <div className='left'>
          <h1>Khách hàng tiềm năng</h1>
        </div>
      </div>
      <div className='bottom-data'>
        <div className='orders'>
          <DataTable 
            value={filterRole} 
            paginator 
            rows={10} 
            dataKey='id' 
            loading={loading} 
            emptyMessage='No users found.'
            header={header}
            globalFilter={globalFilter}
          >
            <Column sortable field='name' header='Khách hàng' />
            {!isSmallScreen && <Column sortable field='email' header='Email' />}
            {!isSmallScreen && <Column sortable field='phone' header='Số điện thoại' />}
            {!isSmallScreen &&<Column sortable field='address' header='Địa chỉ' />}
            <Column sortable field='role' header='Vai trò' body={roleBodyTemplate} />
          </DataTable>
        </div>
      </div>
    </div>
  );
};

export default Page;
