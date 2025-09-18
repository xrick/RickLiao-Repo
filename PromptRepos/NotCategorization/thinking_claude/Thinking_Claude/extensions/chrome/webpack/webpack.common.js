import path from "path"
import { fileURLToPath } from "url"

import CopyPlugin from "copy-webpack-plugin"
import MiniCssExtractPlugin from "mini-css-extract-plugin"

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

export default {
  entry: {
    // popup: path.resolve(__dirname, '..', 'src', 'popup', 'index.tsx'), //popup is not being developed yet
    background: path.resolve(__dirname, "..", "src", "background", "index.ts"),
    content: path.resolve(__dirname, "..", "src", "content", "index.ts"),
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: [
          {
            loader: "ts-loader",
            options: {
              configFile: path.resolve(__dirname, "..", "tsconfig.json"),
              transpileOnly: true,
            },
          },
        ],
        exclude: /node_modules/,
      },
      {
        test: /\.css$/,
        use: [
          MiniCssExtractPlugin.loader,
          {
            loader: "css-loader",
            options: {
              importLoaders: 1,
            },
          },
          "postcss-loader",
        ],
      },
    ],
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"],
    alias: {
      "@": path.resolve(__dirname, "..", "src"),
    },
  },
  output: {
    path: path.resolve(__dirname, "..", "dist"),
    filename: "[name].js",
    clean: true,
  },
  plugins: [
    new MiniCssExtractPlugin(),
    new CopyPlugin({
      patterns: [
        {
          from: path.resolve(__dirname, "..", "public"),
          to: path.resolve(__dirname, "..", "dist"),
          globOptions: {
            ignore: ["**/*.css"], // Ignore CSS files
            patterns: ["**/*.{html,json,png,svg,ico}"], // Only copy specific file types
          },
        },
      ],
    }),
  ],
}
