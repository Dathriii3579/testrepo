import React, { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { motion, AnimatePresence } from "framer-motion";

const profiles = [
  { id: 1, name: "Alice", image: "https://via.placeholder.com/150" },
  { id: 2, name: "Bob", image: "https://via.placeholder.com/150" },
  { id: 3, name: "Charlie", image: "https://via.placeholder.com/150" },
];

export default function SwipeApp() {
  const [index, setIndex] = useState(0);
  const [leftCount, setLeftCount] = useState(0);
  const [rightCount, setRightCount] = useState(0);

  const handleSwipe = (direction) => {
    if (direction === "left") setLeftCount(leftCount + 1);
    else setRightCount(rightCount + 1);
    setIndex((prev) => (prev + 1 < profiles.length ? prev + 1 : 0));
  };

  const profile = profiles[index];

  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-6">
      <div className="text-xl font-bold">Tinder Clone</div>
      <AnimatePresence>
        <motion.div
          key={profile.id}
          initial={{ opacity: 0, x: 100 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -100 }}
          transition={{ duration: 0.3 }}
        >
          <Card className="w-64 shadow-2xl rounded-2xl">
            <CardContent className="p-4 flex flex-col items-center">
              <img src={profile.image} alt={profile.name} className="rounded-full w-32 h-32 mb-4" />
              <div className="text-lg font-medium">{profile.name}</div>
            </CardContent>
          </Card>
        </motion.div>
      </AnimatePresence>
      <div className="flex gap-4">
        <Button onClick={() => handleSwipe("left")} className="bg-red-500 hover:bg-red-600 text-white">
          Swipe Left
        </Button>
        <Button onClick={() => handleSwipe("right")} className="bg-green-500 hover:bg-green-600 text-white">
          Swipe Right
        </Button>
      </div>
      <div className="mt-4 text-center">
        <p>Left Swipes: {leftCount}</p>
        <p>Right Swipes: {rightCount}</p>
      </div>
    </div>
  );
}
