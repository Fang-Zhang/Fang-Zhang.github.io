import Head from "next/head";
import Link from "next/link";

export default function Portfolios({ posts }) {
  return (
    <div>
      <Head>
        <title>Fang Zhang</title>
      </Head>
      <div style={{ textAlign: "start", margin: "2rem", fontSize: "1.5rem" }}>
        I worked in IoT system development and project management in China for
        10 years.
        <br />
        When I came to New Zealand since 2019, I started to understand and
        involve MERN(MongoDB, ExpressJS, ReactJS and NodeJS) Full-Stack
        development around Web3.0.
      </div>
      <br />
      <div style={{ textAlign: "start", margin: "2rem", fontSize: "1.5rem" }}>
        E-mail: <a style={{ color: "blue"}}>walter.zhangfang@gmail.com</a>
        <br />
        People also could get touch with me through:
        - 
        <Link href="https://twitter.com/WalterFangZhang">
          <a target="_blank" style={{ color: "blue" }}>
            Twitter
          </a>
        </Link>
         - or - 
        <Link href="https://github.com/Fang-Zhang">
          <a target="_blank" style={{ color: "blue" }}>
            Github
          </a>
        </Link>
         - or - 
        <Link href="https://www.linkedin.com/in/fang-zhang-ba76aaa4/">
          <a target="_blank" style={{ color: "blue" }}>
            Linkedin
          </a>
        </Link>
        - of course, and - 
        <Link href="/">
          <a target="_blank" style={{ color: "blue" }}>
            This Blog
          </a>
        </Link>
      </div>
      <br />
      <div style={{ textAlign: "start", margin: "2rem", fontSize: "1.5rem" }}>
        <h3>Mobile APP Dev: (React Native)</h3>
        <Link href="https://apps.apple.com/nz/app/ship2u/id1582385021">
          <a target="_blank" style={{ color: "blue" }}>
            Ship2U
          </a>
        </Link>
        : Corporation with the AionTech backend platform, it provides parcels and orders information of customers.
      </div>
      <br />
      <div style={{ textAlign: "start", margin: "2rem", fontSize: "1.5rem" }}>
        <h3> Full-Stack Web Dev: </h3>
        <Link href="https://mdearth.io/">
          <a target="_blank" style={{ color: "blue" }}>
            MDearth.io
          </a>
        </Link>
        : Help the local artists to touch with the Web3.0 world.
        <br />
        <Link href="rainbow-profiterole-c64be5.netlify.app">
          <a target="_blank" style={{ color: "blue" }}>
            Memories
          </a>
        </Link>
        : Showing some memories through the couple of lovely pictures.
        <br />
        <Link href="https://aionlogistech.com/">
          <a target="_blank" style={{ color: "blue" }}>
            AionTech
          </a>
        </Link>
        : Landing page of the AionTech which provides logistics SaaS services.
        <br />
        <Link href="https://sllivings.co.nz/">
          <a target="_blank" style={{ color: "blue" }}>
            SL Living
          </a>
        </Link>
        : Landing page of the Sllivings.
      </div>
      <br />
      <div style={{ textAlign: "start", margin: "2rem", fontSize: "1.5rem" }}>
        <h3>IoT Ecosystem Dev: </h3>
        <Link href="https://www.qivicon.com/en/">
          <a target="_blank" style={{ color: "blue" }}>
            Smart Home Ecosystem
          </a>
        </Link>
        : Ecosystem for the IoT in Germany.
        <br />
        <Link href="https://consumer.huawei.com/ph/offer/category/smart-home/">
          <a target="_blank" style={{ color: "blue" }}>
            Huawei Smart Home
          </a>
        </Link>
        : Ecosystem for the IoT in China.
      </div>
    </div>
  );
}