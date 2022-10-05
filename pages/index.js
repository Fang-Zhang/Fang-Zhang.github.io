// import fs from 'fs'
// import path from 'path'
// import matter from 'gray-matter'
// import Head from 'next/head'
// import Link from 'next/link'
// import Post from '../components/Post'
// import { sortByDate } from '../utils'

// export default function Home({ posts }) {
//   return (
//     <div>
//       <Head>
//         <title>Fang Zhang</title>
//       </Head>
//       <div>I worked in IoT system development and project management in China for 10 years. When I came to New Zealand since 2019, I started to understand and involve MERN(MongoDB, ExpressJS, ReactJS and NodeJS) Full-Stack development.</div>
//       <br/>
//       <div>
//         <h3>Mobile Dev: (React Native)</h3>
//         <Link href="https://apps.apple.com/nz/app/ship2u/id1582385021">
//           <a target="_blank" style={{color: "blue"}}>Ship2U</a>
//         </Link>
//         : Corporation with the Ship2U platform, it provides information about customers.
//       </div>
//       <br/>
//       <div>
//         <h3> MERN Full-Stack Dev: </h3>
//         <Link href="rainbow-profiterole-c64be5.netlify.app">
//           <a target="_blank" style={{color: "blue"}}>Memories</a>
//         </Link>
//         : Showing some memories through the couple of lovely pictures.
//         <br/>

//         <Link href="https://aionlogistech.com/">
//           <a target="_blank" style={{color: "blue"}}>AionTech</a>
//         </Link>
//         : Landing page of the AionTech which provides logistics SaaS services.
//         <br/>

//         <Link href="https://sllivings.co.nz/">
//           <a target="_blank" style={{color: "blue"}}>SL Living</a>
//         </Link>
//         : Landing page of the Sllivings.
//       </div>
//       <br/>
//       <div>
//         <h3>IoT Dev: </h3>
//         <Link href="https://www.qivicon.com/en/">
//           <a target="_blank" style={{color: "blue"}}>Smart Home ecosystem</a>
//         </Link>
//         : Ecosystem for the IoT in Germany.
//       </div>
//       <div className='posts'>
//         {posts.map((post, index) => (
//           <Post key={index} post={post} />
//         ))}
//       </div>
//     </div>
//   )
// }

// export async function getStaticProps() {
//   // Get files from the posts dir
//   const files = fs.readdirSync(path.join('posts'))

//   // Get slug and frontmatter from posts
//   const posts = files.map((filename) => {
//     // Create slug
//     const slug = filename.replace('.md', '')

//     // Get frontmatter
//     const markdownWithMeta = fs.readFileSync(
//       path.join('posts', filename),
//       'utf-8'
//     )

//     const { data: frontmatter } = matter(markdownWithMeta)

//     return {
//       slug,
//       frontmatter,
//     }
//   })

//   return {
//     props: {
//       posts: posts.sort(sortByDate),
//     },
//   }
// }

import fs from "fs";
import path from "path";
import matter from "gray-matter";
import { NextSeo } from "next-seo";
import Post from "../components/Post";
import Banner from "../components/Banner";
import Sidebar from "../components/Sidebar";
import { sortByDate, slugify, ImageUrl } from "../utils";

export default function Home({ posts }) {
  return (
    <div>
      <NextSeo
        title="Welcome to my blog home page"
        description="Build nextjs blog website with Markdown, sitemap, serachbar, category, tag and SEO support"
        openGraph={{
          url: "http://fang-zhang.github.io",
          title: "Welcome to my blog home page",
          description:
            "Build nextjs blog website with Markdown, sitemap, serachbar, category, tag and SEO support",
          images: [
            {
              url: `${ImageUrl("banner.png")}`,
              width: 1224,
              height: 724,
              alt: "banner",
              type: "image/jpeg",
            },
          ],
          site_name: "Fang Zhang",
        }}
      />
      {/* <Banner /> */}
      <div className="container">
        <div className="row">
          <div className="col-lg-8">
            {posts.map((post, index) => (
              <Post key={index} post={post} />
            ))}
          </div>

          <Sidebar />
        </div>
      </div>
    </div>
  );
}

export async function getStaticProps() {
  // Get files from the posts dir
  const files = fs.readdirSync(path.join("posts"));

  // Get slug and frontmatter from posts
  const tempPosts = files.map((filename) => {
    // Create slug
    const slug = filename.replace(".md", "");

    // Get frontmatter
    const markdownWithMeta = fs.readFileSync(
      path.join("posts", filename),
      "utf-8"
    );

    const { data: frontmatter } = matter(markdownWithMeta);

    if (frontmatter.draft === false) {
      return {
        slug,
        frontmatter,
      };
    } else {
      return null;
    }
  });

  //  remove null in tempPosts
  const posts = tempPosts.filter((post) => {
    return post && post;
  });
  const jsonString = JSON.stringify(posts);
  fs.writeFileSync("./search.json", jsonString, (err) => {
    if (err) {
      console.log("Error writing file", err);
    } else {
      console.log("Successfully wrote file");
    }
  });

  return {
    props: {
      posts: posts.sort(sortByDate),
    },
  };
}
