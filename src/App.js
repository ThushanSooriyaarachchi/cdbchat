import {
  useState,
  useEffect,
  useRef,
  useCallback,
  useLayoutEffect,
} from 'react';
import { BiPlus, BiSend, BiSolidUserCircle } from 'react-icons/bi';
import { MdOutlineArrowLeft, MdOutlineArrowRight } from 'react-icons/md';
import axios from "axios";

// Create axios instance with base URL
const apiClient = axios.create({
  baseURL: 'http://127.0.0.1:8000', // Update with your API URL
  headers: {
    'Content-Type': 'application/json',
  },
});

// API service functions
export const ApiService = {
  // Process query
  processQuery: async (query, userId = null) => {
    try {
      const response = await apiClient.post('/process-query', {
        query,
        user_id: userId
      });
      return response.data;
    } catch (error) {
      console.error('Error processing query:', error);
      throw error;
    }
  },

  // Health check
  checkHealth: async () => {
    try {
      const response = await apiClient.get('/health');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }
};

function App() {
  const [text, setText] = useState('');
  const [currentQuery, setCurrentQuery] = useState('');
  const [message, setMessage] = useState(null);
  const [previousChats, setPreviousChats] = useState([]);
  const [localChats, setLocalChats] = useState([]);
  const [currentTitle, setCurrentTitle] = useState(null);
  const [isResponseLoading, setIsResponseLoading] = useState(false);
  const [errorText, setErrorText] = useState('');
  const [isShowSidebar, setIsShowSidebar] = useState(false);
  const scrollToLastItem = useRef(null);

  const createNewChat = () => {
    setMessage(null);
    setText('');
    setCurrentTitle(null);
  };

  const backToHistoryPrompt = (uniqueTitle) => {
    setCurrentTitle(uniqueTitle);
    setMessage(null);
    setText('');
  };

  const toggleSidebar = useCallback(() => {
    setIsShowSidebar((prev) => !prev);
  }, []);

  const submitHandler = async (e) => {
    e.preventDefault();
    if (!text) return;

    // Save the current query before it gets cleared
    const queryText = text;
    setCurrentQuery(queryText);
    
    // Set title if it's a new chat
    if (!currentTitle) {
      setCurrentTitle(queryText);
    }

    setErrorText('');

    try {
      setIsResponseLoading(true);
      
      // First, add the user message to the chat
      const newUserChat = {
        title: currentTitle || queryText,
        role: 'user',
        content: queryText,
      };
      
      setPreviousChats(prevChats => [...prevChats, newUserChat]);
      setLocalChats(prevChats => [...prevChats, newUserChat]);
      
      // Using the ApiService to connect to FastAPI
      const data = await ApiService.processQuery(queryText);
      
      // Process the response
      if (data.status !== 'success') {
        setErrorText(data.detail || 'Error processing your request');
      } else {
        setErrorText('');
        
        // Add the assistant's response to the chat
        const responseMessage = {
          title: currentTitle || queryText,
          role: 'assistant',
          content: data.result,
        };
        
        setPreviousChats(prevChats => [...prevChats, responseMessage]);
        setLocalChats(prevChats => [...prevChats, responseMessage]);
        
        // Update localStorage
        const updatedChats = [...localChats, newUserChat, responseMessage];
        localStorage.setItem('previousChats', JSON.stringify(updatedChats));
        
        // Clear the input field after successful response
        setText('');
        
        // Scroll to the latest message
        setTimeout(() => {
          scrollToLastItem.current?.lastElementChild?.scrollIntoView({
            behavior: 'smooth',
          });
        }, 1);
      }
    } catch (error) {
      console.error('Error processing query:', error);
      setErrorText(error.response?.data?.detail || 'Failed to process your request. Please try again.');
    } finally {
      setIsResponseLoading(false);
    }
  };

  // Check API health on component mount
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        await ApiService.checkHealth();
        setErrorText('');
      } catch (error) {
        setErrorText('API service is currently unavailable');
        console.error('API health check failed:', error);
      }
    };
    
    checkApiHealth();
  }, []);

  useLayoutEffect(() => {
    const handleResize = () => {
      setIsShowSidebar(window.innerWidth <= 640);
    };
    handleResize();

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  useEffect(() => {
    const storedChats = localStorage.getItem('previousChats');

    if (storedChats) {
      setLocalChats(JSON.parse(storedChats));
    }
  }, []);

  const currentChat = (localChats || previousChats).filter(
    (prevChat) => prevChat.title === currentTitle
  );

  const uniqueTitles = Array.from(
    new Set(previousChats.map((prevChat) => prevChat.title).reverse())
  );

  const localUniqueTitles = Array.from(
    new Set(localChats.map((prevChat) => prevChat.title).reverse())
  ).filter((title) => !uniqueTitles.includes(title));

  return (
    <>
      <div className='container'>
        <section className={`sidebar ${isShowSidebar ? 'open' : ''}`}>
          <div className='sidebar-header' onClick={createNewChat} role='button'>
            <BiPlus size={20} />
            <button>New Chat</button>
          </div>
          <div className='sidebar-history'>
            {uniqueTitles.length > 0 && previousChats.length !== 0 && (
              <>
                <p>Ongoing</p>
                <ul>
                  {uniqueTitles?.map((uniqueTitle, idx) => {
                    const listItems = document.querySelectorAll('li');

                    listItems.forEach((item) => {
                      if (item.scrollWidth > item.clientWidth) {
                        item.classList.add('li-overflow-shadow');
                      }
                    });

                    return (
                      <li
                        key={idx}
                        onClick={() => backToHistoryPrompt(uniqueTitle)}
                      >
                        {uniqueTitle}
                      </li>
                    );
                  })}
                </ul>
              </>
            )}
            {localUniqueTitles.length > 0 && localChats.length !== 0 && (
              <>
                <p>Previous</p>
                <ul>
                  {localUniqueTitles?.map((uniqueTitle, idx) => {
                    const listItems = document.querySelectorAll('li');

                    listItems.forEach((item) => {
                      if (item.scrollWidth > item.clientWidth) {
                        item.classList.add('li-overflow-shadow');
                      }
                    });

                    return (
                      <li
                        key={idx}
                        onClick={() => backToHistoryPrompt(uniqueTitle)}
                      >
                        {uniqueTitle}
                      </li>
                    );
                  })}
                </ul>
              </>
            )}
          </div>
          <div className='sidebar-info'>
            <div className='sidebar-info-user'>
              <BiSolidUserCircle size={20} />
              <p>User</p>
            </div>
          </div>
        </section>

        <section className='main'>
          {!currentTitle && (
            <div className='empty-chat-container'>
              <img
                src='/assests/CDB_logo.png'
                width={45}
                height={45}
                alt='ChatCDB'
              />
              <h1>CDB CHAT</h1>
              <h3>Hello CDB, How can I help you today?</h3>
            </div>
          )}

          {isShowSidebar ? (
            <MdOutlineArrowRight
              className='burger'
              size={28.8}
              onClick={toggleSidebar}
            />
          ) : (
            <MdOutlineArrowLeft
              className='burger'
              size={28.8}
              onClick={toggleSidebar}
            />
          )}
          <div className='main-header'>
            <ul>
              {currentChat?.map((chatMsg, idx) => {
                const isUser = chatMsg.role === 'user';

                return (
                  <li key={idx} ref={idx === currentChat.length - 1 ? scrollToLastItem : null}>
                    {isUser ? (
                      <div>
                        <BiSolidUserCircle size={28.8} />
                      </div>
                    ) : (
                      <img src='/assests/CDB_logo.png' alt='ChatCDB' />
                    )}
                    {isUser ? (
                      <div>
                        <p className='role-title'>You</p>
                        <p>{chatMsg.content}</p>
                      </div>
                    ) : (
                      <div>
                        <p className='role-title'>ChatCDB</p>
                        <p>{chatMsg.content}</p>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>
          </div>
          <div className='main-bottom'>
            {errorText && <p className='errorText'>{errorText}</p>}
            <form className='form-container' onSubmit={submitHandler}>
              <input
                type='text'
                placeholder='Send a message.'
                spellCheck='false'
                value={isResponseLoading ? 'Processing...' : text}
                onChange={(e) => setText(e.target.value)}
                readOnly={isResponseLoading}
              />
              {!isResponseLoading && (
                <button type='submit'>
                  <BiSend size={20} />
                </button>
              )}
            </form>
            <p>
              ChatCDB can make mistakes. Consider checking important
              information.
            </p>
          </div>
        </section>
      </div>
    </>
  );
}

export default App;