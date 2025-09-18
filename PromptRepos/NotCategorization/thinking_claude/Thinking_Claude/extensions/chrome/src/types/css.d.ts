/** Extension for typescript if we need to import css files
 * what this does is that it tells typescript that the styles object is an object with string keys and values
 * example usage
 *
 * import styles from './styles.css';
 * Now TypeScript knows that styles is an object with string keys and values element.className = styles.someClass;
 *
 */

declare module "*.css" {
  const content: { [className: string]: string }
  export default content
}

declare module "*.scss" {
  const content: { [className: string]: string }
  export default content
}

declare module "*.sass" {
  const content: { [className: string]: string }
  export default content
}

declare module "*.less" {
  const content: { [className: string]: string }
  export default content
}
