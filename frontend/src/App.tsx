import React, { useState }from 'react';
import './App.css';
import DocumentList from './components/Documents';

//message type
type Message = {
  text: String,
  sender: 'ai' | 'user'
  source_1?: string;
  source_2?: string;
  source_3?: string;
  analysis?: string;
  citation?: string;
}
const App = () => {
  
  //newInputValue is what the user has JUST typed into the textbox
  //setNewInputValue is the function to set this into the array
  //('') is the default: empty string
  const [newInputValue, setNewInputValue] = useState('');

  //messages is an array of all Message objects from user and chat
  //setMessages sets the messages into the chat
  const [messages, setMessages] = useState < Message[] > ([])


  //runs whenever user submits form
  const newMessage: React.FormEventHandler = async (e) => {
    //don't refresh everytime you submit a form, otherwise chat history will be lost
    e.preventDefault();

    //add newInputValue to the newMessages list, which is a snapshot of the messages state variable
    const newMessages: Message[] = [...messages, {
      text: newInputValue,
      sender: 'user'
    }];

    const queryToSend = newInputValue;
    //reset user chat box
    setNewInputValue('')

    //Replaces old "newMessages" array, then re render 
    setMessages(newMessages)
    
    let dotCount = 1;
    const interval = setInterval(() => {
      const dots = ".".repeat(dotCount); //how many times do we have to repeat ., .., ..
      setMessages([...newMessages, {
        sender:'ai',
        text: `generating responses${dots}`}
      ])
      dotCount = (dotCount%3) + 1;
    }, 500)


    const response = await fetch("http://localhost:8000/chat", {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({query: queryToSend})
    });
    

    //save real response into messages state variable now
    const data = await response.json()
    clearInterval(interval);
    setMessages([...newMessages, {
      sender: 'ai',
      text: data.answer,
      source_1: data.source_1,
      source_2: data.source_2,
      source_3: data.source_3,
      analysis: data.source_analysis,
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
            //for every object in messages arr, draw the following"

            //this lne specifically discerns between user or ai (left or right side of chat based on .message.ai vs .message.user in App.css)
            <div key={index} className={"message " + message.sender}>
              
              <div>
              {message.text}
              </div>
              
              {/*Only show these if they exist (Truthiness check) */}
              {message.source_1 && <div><strong>Source: "</strong> {message.source_1} <strong>"</strong></div>}

              {message.analysis && <div><strong>Analysis:</strong> {message.analysis}</div>}

              {message.citation && <div><strong>Citation:</strong> {message.citation}</div>}

              {message.source_2 && <div><strong>Other Relevant Sources: "</strong> {message.source_2} <strong>"</strong></div>}

              {message.source_3 && <div><strong>Other Relevant Sources: "</strong> {message.source_3} <strong>"</strong></div>}
            </div>
          ))}
        </div>

        
        <form className="input-form" onSubmit = {newMessage}>
          <textarea  
            placeholder="What is your question?"
            value = {newInputValue}
            //take new string and send it to setNewInputValue, which puts it in newInputValue var. 
            onChange = {e => setNewInputValue(e.currentTarget.value)}
            onKeyDown = {(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                e.currentTarget.form?.requestSubmit();
              }
            }}
          />
          <input type="submit" value="Send" />
        </form>
      </main>
    </div>
  );
};

export default App;