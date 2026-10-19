// Stylesheets imported for their effect (KaTeX's): Bun bundles them into the page.
declare module "*.css";
// Pictures imported as their URL: Bun copies them into the page's bundle under a hashed name.
declare module "*.svg" {
  const url: string;
  export default url;
}
