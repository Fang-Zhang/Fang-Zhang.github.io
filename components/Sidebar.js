import { useState } from "react";
import Link from "next/link";
import Search from "../search.json";
import { slugify } from "../utils";

export default function Sidebar() {
  const [search, setSearch] = useState();
  function findSerach(value) {
    setSearch(value.target.value);
  }

  return (
    <div className="col-lg-4">
      <div className="card mb-4">
        <div className="card-header">
          <h4>Search</h4>
        </div>
        <div className="card-body">
          <div className="input-group">
            <input
              onChange={findSerach}
              className="form-control"
              type="text"
              placeholder="Enter search term..."
              aria-label="Enter search term..."
              aria-describedby="button-search"
            />
            <Link
              href={{
                pathname: "/Search",
                query: { q: search?.toLowerCase() },
              }}
            >
              <a className="btn btn-primary" id="button-search">
                Go!
              </a>
            </Link>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <div className="card-header">
          <h4>Categories</h4>
        </div>
        <div className="card-body">
          <div className="row">
            <div className="col-sm-10">
              <ul className="list-unstyled mb-0">
                {/* {Search?.map((post) => {
                  return post.frontmatter.categories.map((item) => {
                    const slug = slugify(item);

                    return (
                      <Link key={item} href={`/category/${slug}`}>
                        <a>
                          {" "}
                          <li> {item} </li>
                        </a>
                      </Link>
                    );
                  });
                })} */}
                <li>Soft Skills
                  <ul>
                    <Link href="/category/code-fundamentals">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>Be a proper coder</li>
                      </a>
                    </Link>
                    <Link href="/category/basic-skill">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>Glance the outside</li>
                      </a>
                    </Link>
                  </ul>
                </li>
                <li>Information Technology
                  <ul>
                    <Link href="/category/code-fundamentals">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>Code Fundamentals</li>
                      </a>
                    </Link>
                    <Link href="/category/basic-skill">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>FullStack Dev</li>
                      </a>
                    </Link>
                    <Link href="/category/basic-skill">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>CI/CD</li>
                      </a>
                    </Link>
                    <Link href="/category/basic-skill">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>Design Patterns</li>
                      </a>
                    </Link>
                    <Link href="/category/basic-skill">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>Security</li>
                      </a>
                    </Link>
                    <Link href="/category/basic-skill">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>Architecture</li>
                      </a>
                    </Link>
                  </ul>
                </li>
                <li>Book Review
                  <ul>
                    <Link href="/category/basic-skill">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>Fiction</li>
                      </a>
                    </Link>
                    <Link href="/category/basic-skill">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>Non-Fiction</li>
                      </a>
                    </Link>
                  </ul>
                </li>
                <li>Entrepreneur
                  <ul>
                    <Link href="/category/basic-skill">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>Self Publish</li>
                      </a>
                    </Link>
                    <Link href="/category/basic-skill">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>API</li>
                      </a>
                    </Link>
                    <Link href="/category/basic-skill">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>SaaS</li>
                      </a>
                    </Link>
                    <Link href="/category/basic-skill">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>NFT</li>
                      </a>
                    </Link>
                    <Link href="/category/basic-skill">
                      <a target="_blank" style={{ color: "black" }}>
                        <li>Blockchain</li>
                      </a>
                    </Link>
                  </ul>
                </li>
                <Link href="/category/basic-skill">
                  <a target="_blank" style={{ color: "black" }}>
                    <li>Average Life</li>
                  </a>
                </Link>
                <Link href="/category/basic-skill">
                  <a target="_blank" style={{ color: "black" }}>
                    <li>Classical Chinese Culture</li>
                  </a>
                </Link>
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="card mb-4">
        <div className="card-header">
          <h4>Useful Links</h4>
        </div>
        <div className="card-body">
          <ul>
            <Link href="/portfolios/">
              <a target="_blank" style={{ color: "black" }}>
                <li>My Portfolios</li>
              </a>
            </Link>
            <Link href="https://mdearth.io">
              <a target="_blank" style={{ color: "black" }}>
                <li>MDearth</li>
              </a>
            </Link>
            <Link href="https://www.zhouyi.cc/zhouyi/yijing64/">
              <a target="_blank" style={{ color: "black" }}>
                <li>易經64卦</li>
              </a>
            </Link>
            <Link href="/category/basic-skill">
              <a target="_blank" style={{ color: "black" }}>
                <li>Collections for Coding</li>
              </a>
            </Link>
          </ul>
        </div>
      </div>
    </div>
  );
}
