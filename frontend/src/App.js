import React, { useState } from 'react'
import { Header, Button, Form, Image } from 'semantic-ui-react'
import axios from 'axios'
import './App.css'

const App = () => {
  const [input, setInput] = useState('')
  const [imageUrl, setImageUrl] = useState(null)
  const [resultText, setResultText] = useState([])
  const [resultAttribs, setResultAttribs] = useState([])
  const [resultProb, setResultProb] = useState(null)


  const urlList = [
    'https://m.media-amazon.com/images/M/MV5BMDFkYTc0MGEtZmNhMC00ZDIzLWFmNTEtODM1ZmRlYWMwMWFmXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_.jpg',
    'https://m.media-amazon.com/images/M/MV5BM2MyNjYxNmUtYTAwNi00MTYxLWJmNWYtYzZlODY3ZTk3OTFlXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_SY1000_CR0,0,704,1000_AL_.jpg',
    'https://m.media-amazon.com/images/M/MV5BN2EyZjM3NzUtNWUzMi00MTgxLWI0NTctMzY4M2VlOTdjZWRiXkEyXkFqcGdeQXVyNDUzOTQ5MjY@._V1_SY999_CR0,0,673,999_AL_.jpg',
    'https://m.media-amazon.com/images/M/MV5BNWIwODRlZTUtY2U3ZS00Yzg1LWJhNzYtMmZiYmEyNmU1NjMzXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_.jpg',
    'https://m.media-amazon.com/images/M/MV5BNzQzOTk3OTAtNDQ0Zi00ZTVkLWI0MTEtMDllZjNkYzNjNTc4L2ltYWdlXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_SY1000_CR0,0,665,1000_AL_.jpg',
    'https://m.media-amazon.com/images/M/MV5BZjhkMDM4MWItZTVjOC00ZDRhLThmYTAtM2I5NzBmNmNlMzI1XkEyXkFqcGdeQXVyNDYyMDk5MTU@._V1_SY1000_CR0,0,679,1000_AL_.jpg'
  ]

  if (!imageUrl) {
    setImageUrl(urlList[Math.floor(Math.random() * Math.floor(urlList.length))])
  }
  
  //axios.get('http://localhost:8000/ping').then(res => console.log(res.data))

  const handleButtonPress = (event) => {
    event.preventDefault()
    
    const requestJson = {
      'text': 'I really enjoyed the performance of the actors'
    }

    axios
      .post('http://localhost:8000/api/predict/', requestJson)
      .then(response => {
        const prob = (parseFloat(response.data['prob']) * 100).toFixed(2)
        const attributions = response.data['attributions']
        const text = response.data['text']
        console.log('prob', prob)
        console.log('attr', attributions)
        console.log('text', text)
        setResultText(text)
        setResultAttribs(attributions)
        setResultProb(prob)
      })
  }

  const getColor = (number) => {
    if (number > 0) {
      return number
    } else {
      return -1 * ((number - 1) / (0 - 1))
    }
  }

  const handleInputChange = (event) => {
    event.preventDefault()

    setInput(event.target.value)
  }

  return (
    <div className='App'>
        <Header className='Header'>Please give your opinion on this movie</Header>

        <Image src={imageUrl} />
        
        <Form onSubmit={handleButtonPress}>
          <Form.TextArea placeholder='I had doubts about the actors but...' onChange={handleInputChange} />
          <Button type='submit'>Analyse</Button>
        </Form>

        <div className='resultProb'>
          {resultProb
            ? <h2>{resultProb} %</h2>
            : ''
          }
        </div>

        <div className='resultSentence'>
          {resultText.map((word, i) =>
            <a className='resultWord' key={i} style={{ backgroundColor: 'rgb(0, 255, 50)' }}>{word} {getColor(resultAttribs[i])}</a>)}
        </div>
        
      <div className='Footer'>Made by <a href={'https://github.com/anntey'}>Anntey</a></div>
    </div>
  )
}

export default App
