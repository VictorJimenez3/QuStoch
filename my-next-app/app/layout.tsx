// app/layout.tsx
import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { ReactNode } from 'react'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Stock Analytics Dashboard',
  description: 'Capital One-inspired stock market analytics platform',
}

export default function RootLayout({
  children,
}: {
  children: ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-capital-one-dark-blue`}>
        <header className="bg-capital-one-red text-white p-4 shadow-lg">
          <div className="max-w-6xl mx-auto">
            <h1 className="text-2xl font-bold">Stock Analytics</h1>
          </div>
        </header>
        {children}
      </body>
    </html>
  )
}