import React from 'react'
import { useEffect, useState } from 'react';
import { apiDeleteUser, apiGetUsers} from '../../../apis';
import { DataTable } from 'primereact/datatable';
import { Column } from 'primereact/column';
import { InputText } from 'primereact/inputtext';

const Page = () => {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [globalFilter, setGlobalFilter] = useState('');
  const [isSmallScreen, setIsSmallScreen] = useState(false);

  useEffect(() => {
    const handleResize = () => setIsSmallScreen(window.innerWidth < 700);
    window.addEventListener('resize', handleResize);
    handleResize();

    return () => window.removeEventListener('resize', handleResize);
  }, []);


 const fetchUsers = async () => {
  const response = await apiGetUsers();
  if (response.success) {
    const newestCustomers = response.userData
      .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt)) 
      .slice(0, 10); // Lấy 10 khách hàng mới nhất
    setUsers(newestCustomers);
  }
};

  useEffect(() => {
    fetchUsers();
  }, []);
  const filterRole = users
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
          <h1>Khách hàng mới</h1>
        </div>
      </div>
      <div className='bottom-data'>
        <div className='orders'>
          <DataTable 
            value={filterRole} 
            paginator 
            rows={10} 
            dataKey='id' 
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
