import React, { useState }from 'react';
import './App.css';
import DocumentList from './components/Documents';

//message type
type Message = {
  text: String,
  sender: 'ai' | 'user'
  source?: string;
  analysis?: string;
  citation?: string;
}
const App = () => {
  const [newInputValue, setNewInputValue] = useState('');

  const [messages, setMessages] = useState < Message[] > ([])

  const newMessage: React.FormEventHandler = async (e) => {
    e.preventDefault();
    const newMessages: Message[] = [...messages, {
      text: newInputValue,
      sender: 'user'
    }];
    const queryToSend = newInputValue;
    setNewInputValue('')

    const response = await fetch("http://localhost:8000/chat", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({query: queryToSend})
    });
    const data = await response.json()
    setMessages([...newMessages, {
      sender: 'ai',
      text: data.answer,
      source: data.source,
      analysis: data.analysis,
      citation: data.citation,
    }]);
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>BEACH AI Librarian</h1>
      </header>
      <main>
        <DocumentList />
        <div>
          <h2 className="heading">Talk to a ChatBot:</h2>
          {messages.map((message, index) => (
            <p key={index} className={"message " + message.sender}>
              {message.text}
            </p>
          ))}
        </div>

        <form className="input-form" onSubmit = {newMessage}>
          <input type="text"  placeholder="Message"
          value = {newInputValue}
          onChange = {e => setNewInputValue(e.currentTarget.value)}
          />
          <input type="submit" value="Send" />
        </form>
      </main>
    </div>
  );
};

export default App;