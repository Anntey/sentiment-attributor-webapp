import React, { useState } from 'react'
import { Header, Button, Form, Image } from 'semantic-ui-react'
import axios from 'axios'
import './App.css'

const App = () => {
  const [input, setInput] = useState('')
  const [imageUrl, setImageUrl] = useState(null)
  const [result, setResult] = useState(null)

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
      "sepal length (cm)": 7.9,
      "sepal width (cm)": 2.9,
      "petal length (cm)": 6.3,
      "petal width (cm)": 1.8
    }

    axios
      .post('http://localhost:8000/api/predict/', requestJson)
      .then(response => {
        const pred = response.data['Prediced Iris Species']
        setResult(pred)
      })
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

        <div className='Prediction'>
          {result
            ? <h2>{result} %</h2>
            : ''
          }
        </div>
        
      <div className='Footer'>Made by <a href={'https://github.com/anntey'}>Anntey</a></div>
    </div>
  )
}

export default App
