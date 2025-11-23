import React from 'react'
import {ProductPage, Slider, ProductDeal, ProductPageOne,Featured} from '../Index';

const Home = () => {
  return (
    <div>
      <Slider/>
      <Featured/>
      <ProductPage/>
      <ProductDeal/>
      <ProductPageOne/>
    </div>
  )
}

export default Home