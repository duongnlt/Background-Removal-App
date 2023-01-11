import logo from './logo.svg';
import './App.css';
import ImageUpload from './component/ImageUpload';
import {ChakraProvider, Center, Container, Heading, Text, VStack} from '@chakra-ui/react'
function App() {
  return (
    // <div className="App">
    //   <header>
    //     {/* <img src={logo} className="App-logo" alt="logo" /> */}
    //     {/* <p>
    //       Edit <code>src/App.js</code> and save to reload.
    //     </p>
    //     <a
    //       className="App-link"
    //       href="https://reactjs.org"
    //       target="_blank"
    //       rel="noopener noreferrer"
    //     >
    //       Learn React
    //     </a> */}
    //   </header>
    //   <div>
    //     <ImageUpload/>
    //   </div>
      
    // </div>
    <ChakraProvider>
      <Center bg='black' color='white' padding={8}>
        <VStack spacing='7'>
          <Heading>Background Removal App</Heading>
          <Text>Let upload and remove background from your images</Text>
        </VStack>
      </Center>
      <ImageUpload/>
    </ChakraProvider>
  );
}

export default App;
