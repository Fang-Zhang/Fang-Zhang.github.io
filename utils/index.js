// export const sortByDate = (a, b) => {
//     return new Date(b.frontmatter.date) - new Date(a.frontmatter.date)
//   }

import SITE_URL from "../config";

export function sortByDate(a, b) {
  return new Date(b.frontmatter.date) - new Date(a.frontmatter.date);
}

export function slugify(title) {
  var totalTitle = title
    .toLowerCase()
    .trim()
    .replace(/[^\w ]+/g, "")
    .replace(/ +/g, "-");
  // console.log("totalTile:", totalTitle);
  // var removeDuplicates = totalTitle.filter((item, index) => totalTitle.indexOf(item) === index);
  // return removeDuplicates;
  return totalTitle;
}

export function ImageUrl(url) {
  return SITE_URL + url;
}
