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

  const handleButtonPress = async (event) => {
    event.preventDefault()
    
    const requestJson = {
      'text': `I really didn't enjoy the performance of the actors`
    }

    // const requestJson = {
    //   'text': input
    // }

    // axios
    //   .post('http://localhost:8000/api/predict/', requestJson)
    //   .then(response => {
    //     const prob = (parseFloat(response.data['prob']) * 100).toFixed(2)
    //     const attributions = response.data['attributions']
    //     const text = response.data['text']

    //     setResultText(text)
    //     setResultAttribs(attributions)
    //     setResultProb(prob)
    //   })

    const response = await axios.post('http://localhost:8000/api/predict/', requestJson)

    const prob = (parseFloat(response.data['prob']) * 100).toFixed(0)
    const attributions = response.data['attributions']
    const text = response.data['text']

    setResultText(text)
    setResultAttribs(attributions.map(num => getColor(num)))
    setResultProb(prob)
  }


  const getColor = (number) => {
    if (number > 0) {
      const firstDecimal = parseFloat(number.toString()[2])
      const green_to_white = ['#226122',
      '#2b6a2b',
      '#337333',
      '#3c7c3c',
      '#448544',
      '#559755',
      '#67a867',
      '#78ba78',
      '#89cc89'].reverse()
      return green_to_white[firstDecimal]
    } else {
      const firstDecimal = parseFloat((-1 * number).toString()[2])
      const red_to_white = ['#feb8b8',
      '#f4abab',
      '#ea9e9e',
      '#e09191',
      '#d68585',
      '#c16b6b',
      '#ad5151',
      '#993838',
      '#851e1e']
      .reverse()
      return red_to_white[firstDecimal]
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
            ? <h2>{resultProb} % positive</h2>
            : ''
          }
        </div>

        <div className='resultSentence'>
          {resultText.map((word, i) =>
            <a className='resultWord' key={i} style={{ backgroundColor: `${resultAttribs[i]}` }}>{word}</a>)}
        </div>
        
      <div className='Footer'>Made by <a href={'https://github.com/anntey'}>Anntey</a></div>
    </div>
  )
}

export default App
