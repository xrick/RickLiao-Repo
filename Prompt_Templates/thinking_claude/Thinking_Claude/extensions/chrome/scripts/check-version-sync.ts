#!/usr/bin/env node
import fs from "fs"
import path from "path"
import { fileURLToPath } from "url"

interface PackageJson {
  version: string
  [key: string]: unknown
}

interface ManifestJson {
  manifest_version: number
  name: string
  version: string
  description: string
  [key: string]: unknown
}

const __filename: string = fileURLToPath(import.meta.url)
const __dirname: string = path.dirname(__filename)

const packageJsonPath: string = path.resolve(__dirname, "..", "package.json")
const manifestJsonPath: string = path.resolve(
  __dirname,
  "..",
  "public",
  "manifest.json"
)

try {
  const packageJson: PackageJson = JSON.parse(
    fs.readFileSync(packageJsonPath, "utf8")
  )
  const manifestJson: ManifestJson = JSON.parse(
    fs.readFileSync(manifestJsonPath, "utf8")
  )

  if (packageJson.version !== manifestJson.version) {
    console.error("\x1b[31mError: Version mismatch!\x1b[0m")
    console.error(`package.json version: ${packageJson.version}`)
    console.error(`manifest.json version: ${manifestJson.version}`)
    console.error(
      "\nPlease update the version in public/manifest.json to match package.json"
    )
    process.exit(1)
  }

  console.log("\x1b[32mVersions are in sync ✓\x1b[0m")
  process.exit(0)
} catch (error: unknown) {
  console.error(
    "\x1b[31mError checking version sync:\x1b[0m",
    error instanceof Error ? error.message : error
  )
  process.exit(1)
}
