import React, { useEffect, useState } from 'react';
import { FaSearch } from 'react-icons/fa';
import { logout } from '../../store/user/userSlice';
import withBaseComponents from '../../hocs/withBaseComponents';
import { FaMessage } from 'react-icons/fa6';
import { IoIosLogOut } from "react-icons/io";
import { BiNotification } from 'react-icons/bi';
import { apiGetNotifications, apiGetUsers } from '../../apis';
import NotificationAdmin from '../public/notifice/NotificeAdmin';

const HeaderAdmin = ({dispatch, navigate}) => {
  const [showNotification, setShowNotification] = useState(false);
  const [notifications, setNotifications] = useState([]); 
  const [users, setUsers] = useState([]);
  const fetchNotification = async () => {
        try {
           const response = await apiGetNotifications();
           if(response.success && response.data) {
              setNotifications(response.data); 
           }
        } catch (error) {
           console.error('Error fetching notifications:', error);
        }
     }
  const fetchUsers = async () => {
    try{
      const response = await apiGetUsers();
      if(response.success && response.userData?.length > 0){
        setUsers(response.userData);
        console.log(response.userData)
      }
    }catch (error) {
           console.error('Error fetching notifications:', error);
        }
  }   
  useEffect(() => {
    fetchNotification();
    fetchUsers();
  },[])   
  const handleLogout = () => {
    dispatch(logout());  
    navigate('/'); 
  };
  return (
   
      <nav>
      <i className='bx bx-menu'></i>
      <form action='#'>
        <div className='form-input'>
          <input type='search' placeholder='Search...' />
          <button className='search-btn' type='submit'><FaSearch/></button>
        </div>
      </form>
      <a href={'/admin/message'} style={{cursor: 'pointer', color: 'blue'}} ><FaMessage/></a>
      <span style={{cursor: 'pointer'}} onClick={() => handleLogout()}><IoIosLogOut /></span>
      <span style={{cursor: 'pointer'}} onClick={() => setShowNotification(prev => !prev)}><BiNotification /></span>
      {showNotification && (
         <div className='notification-template-admin shadow '>
            <NotificationAdmin
              users={users}
               notification={notifications}
            />
         </div>
      )}
    </nav>
    
  );
}

export default withBaseComponents(HeaderAdmin);
