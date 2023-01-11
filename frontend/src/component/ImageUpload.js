import React, {Component} from 'react'
import {Container, Button, SimpleGrid, Image, HStack, Input} from '@chakra-ui/react'


class ImageUpload extends Component{
    constructor(props){
        super(props)

        this.state = {
            currentFile: undefined,
            previewImage: undefined,
            resultImage: undefined,
            progress: 0,
            message: "",
        }

    }
    selectFile = event => {
        this.setState({
          currentFile: event.target.files[0],
          previewImage: URL.createObjectURL(event.target.files[0]),
          progress: 0,
          message: ""
        });
    }

    removePreview = event =>{
      this.setState({
        currentFile: undefined,
        previewImage: undefined,
        resultImage: undefined
        // progress: 0,
        // message: ""
      });
    }

    upload = ()=>{
       const formData = new FormData()
       formData.append(
          'image',
          this.state.currentFile,
          this.state.currentFile.name
       )
       const requestOptions = {
        method: 'POST',
        // headers: { 'Content-Type': 'multipart/form-data' }, // DO NOT INCLUDE HEADERS
        body: formData
        };
      fetch('http://0.0.0.0:8000/predict', requestOptions)
        .then(response => response.blob())
        .then(response => {
          console.log('response')
          console.log(response)
          const objectURL = URL.createObjectURL(response);
          this.setState({
            resultImage: objectURL
          })
            });
    }

    render(){
        const {
            currentFile,
            previewImage,
            progress,
            message,
            resultImage
          } = this.state;
        return (
        <Container padding={10} >
          <Container>
            <HStack spacing={8}> 
              {/* <input type="file" onChange={this.selectFile} /> */}
              <Input type='file' onChange={this.selectFile} variant='flushed' onClick={e => (e.target.value = null)}/>
              <Button size='lg' colorScheme='red' isDisabled={!currentFile} onClick={this.upload}>Upload</Button>
            </HStack>
          </Container>
          <SimpleGrid columns={2}>
              {previewImage && (
                  <Image padding={5} borderRadius={10} src={previewImage}  objectFit='cover' alt='not found'/>
                
              )}
              {resultImage && (
                  <Image padding={5}  borderRadius={10} src={resultImage}  objectFit='cover' alt='no result'/>
              
              )}
          </SimpleGrid>
          <Container margin='0 auto' width='90px'>
          {previewImage && (
            <Button onClick={this.removePreview}>Remove</Button>
          )}
          </Container>
        </Container>
      //   <div>
      //       <label className="btn btn-default p-0">
      //         <input type="file" onChange={this.selectFile} />
      //       </label>
      //       <div className="col-4">
      //       <button
      //         className="btn btn-success btn-sm"
      //         disabled={!currentFile}
      //         onClick={this.upload}
      //       >
      //         Upload
      //       </button>
      //     </div>
        
      //   {previewImage && (
      //   <div>
      //   <img alt="not found" width={"320px"} src={previewImage} />
      //   <br />
      //   </div>
      // )}
      //         {resultImage && (
      //   <div>
      //   <img alt="not found" width={"320px"} src={resultImage} />

      //   </div>
      // )}
      // {  previewImage && (
      //   <button onClick={this.removePreview}>Remove</button>
      // )
      // }
      //   </div>
            
        )
    }
}

export default ImageUpload