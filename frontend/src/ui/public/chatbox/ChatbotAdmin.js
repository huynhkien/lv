import React, { useState, useEffect } from 'react';
import { MdCancel } from 'react-icons/md';
import { FaArrowCircleRight } from "react-icons/fa";
import { sendMessageAdmin } from '../../../apis/chatbox';
import userAvatar from '../../../assets/img/logo/9187604.png'; 
import botAvatar from '../../../assets/img/logo/images.png'

const ChatBotAdmin = ({ setShowOption }) => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');

  useEffect(() => {
    setMessages([{ type: 'bot', content: 'Xin chào! Tôi có thể giúp gì cho bạn?' }]);
  }, []);

  const handleSendMessage = async () => {
    if (inputMessage.trim() === '') return;

    const userMessage = {
      type: 'user',
      content: inputMessage,
    };
    setMessages(prevMessages => [...prevMessages, userMessage]);

    try {
        const response = await sendMessageAdmin(inputMessage);
        setMessages(prevMessages => [...prevMessages, { 
          type: 'bot', 
          content: response.response 
        }]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prevMessages => [...prevMessages, { 
        type: 'error', 
        content: 'Xin lỗi, đã xảy ra lỗi khi xử lý tin nhắn của bạn.' 
      }]);
    }
    setInputMessage('');
  };

  return (
      <div className="chatbot-container">
        <div className="chatbot-header">
          <span className='message-header-icon'>
            <span>ChatBot</span>
          </span>
          <span className='chatbot-header-icon' onClick={setShowOption}>
            <MdCancel/>
          </span>
        </div>
        <div className="chatbot-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`chat-message ${msg.type}-message`}>
              <img 
                src={msg.type === 'user' ? userAvatar : botAvatar} 
                alt="Avatar" 
                className="avatar"
              />
              <div>
                {msg.content}
              </div>
            </div>
          ))}
        </div>

        <div className="chatbot-body">
          <div className='chatbot-input-container'>
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Câu hỏi..."
            className="chatbot-input"
          />
          <button onClick={handleSendMessage} className="send-button">
            <FaArrowCircleRight/>
          </button>
          </div>
        </div>
    </div>
  );
};

export default ChatBotAdmin;
