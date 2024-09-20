import { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [message, setMessage] = useState<string>('');

  useEffect(() => {
    axios.get('http://localhost:5000/api/hello')
      .then(response => setMessage(response.data.message))
      .catch(error => console.error('Error:', error));
  }, []);

  return (
    <div className="App">
      <h1 className="text-3xl font-bold underline">
        NBA Fantasy Projector
      </h1>
      <p>{message}</p>
    </div>
  );
}

export default App;
