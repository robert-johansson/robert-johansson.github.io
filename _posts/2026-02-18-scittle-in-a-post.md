---
layout: post
title: "ClojureScript in a Blog Post"
date: 2026-02-18 20:00:00 +0100
categories: [clojurescript]
tags: [scittle, clojurescript, reagent]
share: false
---

Can we run ClojureScript directly inside a blog post? Let's find out.

<div id="app"></div>

<script src="https://unpkg.com/react@18/umd/react.production.min.js" crossorigin></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js" crossorigin></script>
<script src="https://cdn.jsdelivr.net/npm/scittle@0.8.31/dist/scittle.js" type="text/javascript"></script>
<script src="https://cdn.jsdelivr.net/npm/scittle@0.8.31/dist/scittle.reagent.js" type="text/javascript"></script>

<script type="application/x-scittle">
(require '[reagent.core :as r]
         '[reagent.dom :as rdom])

(defonce items (r/atom ["Probabilistic programming" "Machine psychology" "NARS"]))
(defonce new-item (r/atom ""))

(defn app []
  [:div {:style {:padding "1.5em"
                 :border "1px solid #ddd"
                 :border-radius "8px"
                 :max-width "500px"}}
   [:h3 {:style {:margin-top 0}} "Research Topics"]
   [:ul (for [item @items]
          ^{:key item} [:li item])]
   [:div {:style {:display "flex" :gap "0.5em"}}
    [:input {:type "text"
             :value @new-item
             :placeholder "Add a topic..."
             :on-change #(reset! new-item (.. % -target -value))
             :style {:padding "0.4em" :flex 1}}]
    [:button {:on-click (fn []
                          (when (seq @new-item)
                            (swap! items conj @new-item)
                            (reset! new-item "")))
              :style {:padding "0.4em 1em" :cursor "pointer"}}
     "Add"]]])

(rdom/render [app] (.getElementById js/document "app"))
</script>

It works — the component above is a live Reagent app rendered from ClojureScript embedded in this markdown file.
