import React from 'react'
import  {Navigate, Outlet} from 'react-router-dom';
import {SidebarAdmin, HeaderAdmin} from '../Index';
import HeaderList from './HeaderList';
import { useSelector } from 'react-redux';
import { FaRobot } from 'react-icons/fa';
import { useState } from 'react';
import ChatBotAdmin from '../public/chatbox/ChatbotAdmin';

const Admin = () => {
  const {current} = useSelector(state => state.user);
  const [showChatBot, setShowChatBot] = useState(false);
  if (!current) {
    return <Navigate to="/login" replace />;
  }else if(current && current?.role === 2004){
    return <Navigate to="/login" replace />;
  }
  const handleChatBotToggle = () => {
    setShowChatBot(prev => !prev); 
  };
  return (
    <div className='admin-layout'>
        <HeaderList />
        <SidebarAdmin />
        <div className='content'>
          <HeaderAdmin />
          <main>
            <Outlet/>
          </main>
          <div className='chat--item'>
                  <FaRobot
                    onClick={handleChatBotToggle} 
                    className='icon' 
                    size={50} 
                  />
                </div>
                {showChatBot && (
                  <div className='chatBox-template'>
                    <ChatBotAdmin
                      setShowOption={handleChatBotToggle}
                    />
                  </div>
                )}
        </div>
    </div>
  )
}

export default Admin