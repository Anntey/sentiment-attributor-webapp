import React, { useState, useEffect } from 'react'
import './App.css'
import { Header, Segment, Input, Image } from 'semantic-ui-react'

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
  useEffect(() => {
    if (!imageUrl) {
      setImageUrl(urlList[Math.floor(Math.random() * Math.floor(urlList.length))])
    }
  })
  

  // ping backend
  // axios.get('http://localhost:8000/ping').then(res => console.log(res.data))

  // const handleInputChange = (event) => {
  //   event.preventDefault()

  //   const inputFile = event.target.files[0]

  //   if (inputFile) {
  //     setFileUrl(URL.createObjectURL(inputFile))

  //     const formData = new FormData()
  //     formData.append('img', inputFile, inputFile.name)

  //     const config = {
  //       headers: [
  //         { 'Content-Type': 'multipart/form-data' },
  //         { 'Access-Control-Allow-Origin': '*' }
  //       ]
  //     }

  //     axios
  //       .post('http://localhost:8000/predict', formData, config)
  //       .then(response => {
  //         const prob = parseFloat(response.data.prediction)
  //         setPred(prob)
  //       })
  //   }
  // }

  const handleInputChange = (event) => {
    event.preventDefault()

    setInput(event.target.value)
  }

  return (
    <div className='App'>
      <Segment className='Segment'>

        <Header className='Header'>Please give your opinion on this movie</Header>

        <Image src={imageUrl} />

        <Input placeholder='I had doubts about the actors' onChange={handleInputChange}/>

{/* 
        <Button
          className='Input'
          type='file'
          onChange={handleInputChange}
        /> */}
        

        {/* <div className='Prediction'>
        {pred
          ? <h2>Prediction: {(pred * 100).toFixed(3)} %</h2>
          : ''
        }
        </div> */}
        

      </Segment>
      <div className='Footer'>Made by <a href={'https://github.com/anntey'}>Anntey</a></div>
    </div>
  )
}

export default App
